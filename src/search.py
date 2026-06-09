"""Multi-signal email search: FTS5 lexical + vector cosine + cluster centroid."""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import requests
from dotenv import load_dotenv

try:
    from .db import get_connection
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.db import get_connection

load_dotenv()

DB_PATH     = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"

WEIGHT_FTS     = 0.3
WEIGHT_VEC     = 0.5
WEIGHT_CLUSTER = 0.2
NOISE_PENALTY  = -0.1

FTS_LIMIT     = 300
VEC_TOP_K     = 100
CLUSTER_TOP_K = 3


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_query(text: str) -> np.ndarray:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    arr = np.array(resp.json()["embedding"], dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


# ── FTS5 ──────────────────────────────────────────────────────────────────────

def _ensure_fts(conn) -> None:
    try:
        cfg = conn.execute("SELECT v FROM emails_fts_config WHERE k='tokenize'").fetchone()
        if cfg is None or "remove_diacritics" not in cfg[0]:
            conn.execute("DROP TABLE IF EXISTS emails_fts")
    except Exception:
        pass

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS emails_fts
        USING fts5(subject, body_text, content=emails, content_rowid=id,
                   tokenize="unicode61 remove_diacritics 1")
    """)
    conn.commit()

    email_count = conn.execute(
        "SELECT COUNT(*) FROM emails WHERE body_text IS NOT NULL"
    ).fetchone()[0]
    page_count = conn.execute("SELECT COUNT(*) FROM emails_fts_data").fetchone()[0]
    if page_count < max(5, email_count // 200):
        conn.execute("INSERT INTO emails_fts(emails_fts) VALUES('rebuild')")
        conn.commit()


def _sanitize_fts(text: str) -> str:
    words = re.findall(r'\b\w{2,}\b', text.lower())[:12]
    return " OR ".join(f'"{w}"' for w in words) if words else ""


def _fts_search(conn, query: str) -> dict[int, float]:
    fts_q = _sanitize_fts(query)
    if not fts_q:
        return {}
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM emails_fts WHERE emails_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_q, FTS_LIMIT),
        ).fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    ranks = [r[1] for r in rows]
    best, worst = min(ranks), max(ranks)
    if best == worst:
        return {int(r[0]): 1.0 for r in rows}
    return {int(r[0]): (r[1] - worst) / (best - worst) for r in rows}


# ── embeddings matrix ─────────────────────────────────────────────────────────

def _load_embeddings(conn) -> tuple[list[int], np.ndarray, list[int | None]]:
    rows = conn.execute("""
        SELECT e.id, e.embedding, ec.cluster_id
        FROM emails e
        LEFT JOIN (
            SELECT email_id, cluster_id FROM email_clusters WHERE source = 'hdbscan'
        ) ec ON e.id = ec.email_id
        WHERE e.embedding IS NOT NULL
    """).fetchall()
    email_ids   = [r[0] for r in rows]
    cluster_ids = [r[2] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return email_ids, matrix / norms, cluster_ids


# ── vector search ─────────────────────────────────────────────────────────────

def _vector_search(
    email_ids: list[int], norm_matrix: np.ndarray, q: np.ndarray, top_k: int = VEC_TOP_K
) -> dict[int, float]:
    sims = norm_matrix @ q
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return {email_ids[i]: float(max(0.0, sims[i])) for i in top_idx}


# ── cluster search (noise penalty) ───────────────────────────────────────────

def _cluster_search(
    email_ids: list[int],
    norm_matrix: np.ndarray,
    cluster_ids: list[int | None],
    q: np.ndarray,
    top_k: int = CLUSTER_TOP_K,
) -> dict[int, float]:
    members: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        if cid is not None:
            members[cid].append(i)

    result: dict[int, float] = {}
    if members:
        cids = list(members.keys())
        centroid_list = []
        for cid in cids:
            c = norm_matrix[members[cid]].mean(axis=0)
            n = np.linalg.norm(c)
            centroid_list.append(c / n if n > 0 else c)
        sims    = np.stack(centroid_list) @ q
        top_idx = np.argsort(sims)[-top_k:][::-1]
        for rank, idx in enumerate(top_idx):
            score = float(max(0.0, sims[idx])) * [1.0, 0.7, 0.5][rank]
            for i in members[cids[idx]]:
                eid = email_ids[i]
                if eid not in result or result[eid] < score:
                    result[eid] = score

    # noise penalty
    for i, cid in enumerate(cluster_ids):
        if cid is None:
            eid = email_ids[i]
            if eid not in result:
                result[eid] = NOISE_PENALTY

    return result


# ── fetch details ─────────────────────────────────────────────────────────────

def _fetch_details(conn, ids: list[int], scores: dict[int, float]) -> list[dict]:
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT e.id, e.subject, e.from_address, e.date, e.body_snippet,
               e.thread_id, c.label AS cluster_label
        FROM emails e
        LEFT JOIN (SELECT email_id, cluster_id FROM email_clusters WHERE source='hdbscan') ec
               ON e.id = ec.email_id
        LEFT JOIN clusters c ON ec.cluster_id = c.id
        WHERE e.id IN ({ph})
        """,
        ids,
    ).fetchall()
    result = [
        {
            "id":             r["id"],
            "subject":        (r["subject"] or "").strip(),
            "from_address":   r["from_address"] or "",
            "date":           (r["date"] or "")[:10],
            "score":          round(scores.get(r["id"], 0.0), 4),
            "cluster_label":  r["cluster_label"] or "noise",
            "body_snippet":   (r["body_snippet"] or "").strip(),
            "thread_id":      r["thread_id"],
            "expanded":       False,
            "person_expanded": False,
        }
        for r in rows
    ]
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


# ── thread expansion ──────────────────────────────────────────────────────────

def _expand_threads(conn, direct: list[dict]) -> list[dict]:
    thread_ids = {r["thread_id"] for r in direct if r["thread_id"]}
    if not thread_ids:
        return direct
    known_ids = {r["id"] for r in direct}
    tid_ph = ",".join("?" * len(thread_ids))
    did_ph = ",".join("?" * len(known_ids))
    rows = conn.execute(
        f"""
        SELECT e.id, e.subject, e.from_address, e.date, e.body_snippet,
               e.thread_id, c.label AS cluster_label
        FROM emails e
        LEFT JOIN (SELECT email_id, cluster_id FROM email_clusters WHERE source='hdbscan') ec
               ON e.id = ec.email_id
        LEFT JOIN clusters c ON ec.cluster_id = c.id
        WHERE e.thread_id IN ({tid_ph}) AND e.id NOT IN ({did_ph})
        ORDER BY e.date ASC
        """,
        list(thread_ids) + list(known_ids),
    ).fetchall()
    expanded = [
        {
            "id":             r["id"],
            "subject":        (r["subject"] or "").strip(),
            "from_address":   r["from_address"] or "",
            "date":           (r["date"] or "")[:10],
            "score":          None,
            "cluster_label":  r["cluster_label"] or "noise",
            "body_snippet":   (r["body_snippet"] or "").strip(),
            "thread_id":      r["thread_id"],
            "expanded":       True,
            "person_expanded": False,
        }
        for r in rows
    ]
    return direct + expanded


# ── person expansion ──────────────────────────────────────────────────────────

def _parse_addresses(to_json: str | None) -> set[str]:
    if not to_json:
        return set()
    try:
        return {
            a.get("address", "").lower().strip()
            for a in json.loads(to_json)
            if a.get("address", "").strip()
        }
    except Exception:
        return set()


def expand_persons(
    results: list[dict],
    db_path: str = DB_PATH,
    window_days: int = 90,
    min_overlap: int = 2,
    context_threshold: float = 0.75,
) -> tuple[list[dict], set[str]]:
    """Find emails in ±window_days time window sharing ≥min_overlap seed persons,
    filtered by cosine similarity > context_threshold with the centroid of direct hits.

    Returns (person_expanded_list, seed_persons).
    """
    direct = [r for r in results if not r.get("expanded") and not r.get("person_expanded")]
    if not direct:
        return [], set()

    conn = get_connection(db_path)
    direct_ids = [r["id"] for r in direct]
    ph = ",".join("?" * len(direct_ids))
    rows = conn.execute(
        f"SELECT id, from_address, to_addresses, date, embedding FROM emails WHERE id IN ({ph})",
        direct_ids,
    ).fetchall()

    seed_persons: set[str] = set()
    dates: list[str] = []
    centroid_vecs: list[np.ndarray] = []

    for row in rows:
        if row["from_address"]:
            seed_persons.add(row["from_address"].lower().strip())
        seed_persons |= _parse_addresses(row["to_addresses"])
        if row["date"]:
            dates.append(row["date"][:10])
        if row["embedding"]:
            centroid_vecs.append(np.frombuffer(row["embedding"], dtype=np.float32))
    seed_persons.discard("")

    if not seed_persons or not dates:
        conn.close()
        return [], seed_persons

    # centroid of direct-hit embeddings (normalised)
    if centroid_vecs:
        centroid = np.mean(np.stack(centroid_vecs), axis=0)
        c_norm = np.linalg.norm(centroid)
        centroid = centroid / c_norm if c_norm > 0 else centroid
    else:
        centroid = None

    dates.sort()
    t_start = (datetime.fromisoformat(dates[0]) - timedelta(days=window_days)).isoformat() + "Z"
    t_end   = (datetime.fromisoformat(dates[-1]) + timedelta(days=window_days)).isoformat() + "Z"

    known_ids = {r["id"] for r in results}
    seed_list = list(seed_persons)
    ph_seeds = ",".join("?" * len(seed_list))
    ph_known = ",".join("?" * len(known_ids))

    candidates = conn.execute(
        f"""
        SELECT e.id, e.subject, e.from_address, e.to_addresses, e.date,
               e.body_snippet, e.thread_id, e.embedding, c.label AS cluster_label
        FROM emails e
        LEFT JOIN (SELECT email_id, cluster_id FROM email_clusters WHERE source='hdbscan') ec
               ON e.id = ec.email_id
        LEFT JOIN clusters c ON ec.cluster_id = c.id
        WHERE e.date >= ? AND e.date <= ?
          AND e.id NOT IN ({ph_known})
          AND e.from_address IN ({ph_seeds})
        ORDER BY e.date ASC
        """,
        [t_start, t_end] + list(known_ids) + seed_list,
    ).fetchall()

    person_results: list[dict] = []
    n_person_overlap = 0

    for row in candidates:
        email_persons: set[str] = set()
        if row["from_address"]:
            email_persons.add(row["from_address"].lower().strip())
        email_persons |= _parse_addresses(row["to_addresses"])

        overlap = email_persons & seed_persons
        if len(overlap) < min_overlap:
            continue
        n_person_overlap += 1

        # context filter: cosine similarity with centroid of direct hits
        if centroid is not None and row["embedding"]:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            v_norm = np.linalg.norm(vec)
            if v_norm > 0:
                vec = vec / v_norm
            similarity = float(np.dot(vec, centroid))
            if similarity <= context_threshold:
                continue
            person_score = round(similarity, 4)
        else:
            person_score = None  # no embedding → keep but no score

        person_results.append({
            "id":               row["id"],
            "subject":          (row["subject"] or "").strip(),
            "from_address":     row["from_address"] or "",
            "date":             (row["date"] or "")[:10],
            "score":            person_score,
            "cluster_label":    row["cluster_label"] or "noise",
            "body_snippet":     (row["body_snippet"] or "").strip(),
            "thread_id":        row["thread_id"],
            "expanded":         False,
            "person_expanded":  True,
            "matching_persons": sorted(overlap),
        })

    conn.close()

    person_results.sort(key=lambda r: r["score"] or 0, reverse=True)
    print(
        f"  [person expansion: person_overlap={n_person_overlap}"
        f"  context_filtered={n_person_overlap - len(person_results)}"
        f"  kept={len(person_results)}  threshold={context_threshold}]",
        file=sys.stderr,
    )
    return person_results, seed_persons


# ── public search API ─────────────────────────────────────────────────────────

def search(
    query:     str,
    db_path:   str   = DB_PATH,
    top_k:     int   = 20,
    min_score: float = 0.55,
) -> list[dict]:
    """Return top_k emails ranked by FTS + vector + cluster, with thread expansion."""
    conn = get_connection(db_path)
    t0   = time.time()

    q_vec = embed_query(query)

    _ensure_fts(conn)
    fts_scores = _fts_search(conn, query)

    email_ids, norm_matrix, cluster_ids = _load_embeddings(conn)
    vec_scores     = _vector_search(email_ids, norm_matrix, q_vec)
    cluster_scores = _cluster_search(email_ids, norm_matrix, cluster_ids, q_vec)

    all_ids = set(fts_scores) | set(vec_scores) | set(cluster_scores)
    combined = {
        eid: (
            WEIGHT_FTS     * fts_scores.get(eid, 0.0)
            + WEIGHT_VEC     * vec_scores.get(eid, 0.0)
            + WEIGHT_CLUSTER * cluster_scores.get(eid, 0.0)
        )
        for eid in all_ids
    }

    top_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
    direct  = _fetch_details(conn, top_ids, combined)
    direct  = [r for r in direct if r["score"] >= min_score]
    results = _expand_threads(conn, direct)
    conn.close()

    elapsed    = time.time() - t0
    n_expanded = sum(1 for r in results if r["expanded"])
    print(
        f"  [{elapsed:.2f}s  FTS:{len(fts_scores)}  VEC:{len(vec_scores)}"
        f"  CLU:{len(cluster_scores)}  direct:{len(direct)}  thread:+{n_expanded}]",
        file=sys.stderr,
    )
    return results


# ── CLI display ───────────────────────────────────────────────────────────────

def _group_threads(results: list[dict]) -> list[list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    no_thread: list[dict] = []
    for r in results:
        if r["thread_id"]:
            buckets[r["thread_id"]].append(r)
        else:
            no_thread.append(r)

    def _best(g: list[dict]) -> float:
        s = [r["score"] for r in g if r["score"] is not None]
        return max(s) if s else 0.0

    groups = [sorted(g, key=lambda r: r["date"] or "") for g in buckets.values()]
    groups.sort(key=_best, reverse=True)
    return groups + [[r] for r in sorted(no_thread, key=lambda r: r["date"] or "")]


def _print_section(results: list[dict], title: str) -> None:
    direct   = [r for r in results if not r.get("expanded")]
    n_expand = sum(1 for r in results if r.get("expanded"))
    groups   = _group_threads(results)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  Priamych: {len(direct)}  |  thread+: {n_expand}  |  vlakien: {len(groups)}")
    print(f"{'=' * 60}")

    hdr = f"  {'Datum':10}  {'Od':28}  {'Subject':36}  {'Score':5}  Cluster"
    sep = "  " + "-" * 100

    for g_idx, group in enumerate(groups, 1):
        best_subj = next(
            (r["subject"] for r in group if not r.get("expanded")), group[0]["subject"]
        )
        print(f"\n--- vlakno {g_idx} — {best_subj[:55]} ---")
        print(hdr)
        print(sep)
        for r in group:
            marker = "~" if r.get("expanded") else " "
            score  = f"{r['score']:.3f}" if r["score"] is not None else "  ---"
            print(
                f"{marker} {r['date']:10}  {r['from_address'][:27]:28}"
                f"  {r['subject'][:35]:36}  {score:5}  {r['cluster_label'][:22]}"
            )


def _print_persons_section(person_results: list[dict], seed_persons: set[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  PERSON EXPANSION  ({len(person_results)} emailov)")
    print(f"{'=' * 60}")

    print(f"\n  Seed osoby ({len(seed_persons)}):")
    for p in sorted(seed_persons):
        print(f"    {p}")

    if not person_results:
        print("\n  (ziadne emaily presli kontextovym filtrom)")
        return

    print(f"\n  {'Datum':10}  {'Od':28}  {'Subject':36}  {'Sim':5}  {'Zhody':30}  Cluster")
    print("  " + "-" * 120)
    for r in person_results:
        persons_short = ", ".join(r.get("matching_persons", []))[:28]
        sim = f"{r['score']:.3f}" if r["score"] is not None else "  ---"
        print(
            f"  {r['date']:10}  {r['from_address'][:27]:28}"
            f"  {r['subject'][:35]:36}  {sim:5}  {persons_short:30}  {r['cluster_label'][:22]}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-signal email search")
    parser.add_argument("query",            help="Search query")
    parser.add_argument("--top",            type=int,   default=20,   help="Direct results (default 20)")
    parser.add_argument("--min-score",      type=float, default=0.55, help="Min score threshold (default 0.55)")
    parser.add_argument("--db",             default=DB_PATH,          help="Path to SQLite DB")
    parser.add_argument("--expand-persons", action="store_true",      help="Expand results by seed persons")
    args = parser.parse_args()

    results = search(args.query, db_path=args.db, top_k=args.top, min_score=args.min_score)

    _print_section(results, f'Vysledky: "{args.query}"  (min_score={args.min_score})')

    if args.expand_persons:
        t0 = time.time()
        person_results, seed_persons = expand_persons(results, db_path=args.db)
        print(f"  [person expansion: {time.time()-t0:.2f}s  kandidatov:{len(person_results)}]",
              file=sys.stderr)
        _print_persons_section(person_results, seed_persons)


if __name__ == "__main__":
    main()
