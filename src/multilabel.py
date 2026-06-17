"""Multi-label topic profiling via mean+2std threshold on centroid similarities.

Strategy: for each email, compute cosine similarity to all cluster centroids,
then select clusters where sim >= mean + 2*std. Fallback: if nothing is
selected, assign top-1 cluster and mark low_confidence=1.

Writes to email_topics table — does NOT modify email_clusters.

Usage: python -m src.multilabel
"""

import io
import os
import sqlite3
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.environ.get("DB_PATH", "data/emails.db")


def _fix_stdout():
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS email_topics (
            email_id       INTEGER NOT NULL,
            cluster_id     INTEGER NOT NULL,
            similarity     REAL    NOT NULL,
            rank           INTEGER NOT NULL,
            low_confidence INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (email_id, cluster_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_et_email   ON email_topics(email_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_et_cluster ON email_topics(cluster_id)")
    conn.commit()


def _load_centroids(conn):
    """Return (centroid_matrix, cluster_ids, labels).

    centroid_matrix: np.ndarray (n_clusters, dim), L2-normalised
    cluster_ids:     list[int] — index i → cluster_ids[i]
    labels:          dict[int, str]
    """
    labels = {r[0]: r[1] for r in conn.execute(
        "SELECT id, label FROM clusters WHERE label IS NOT NULL"
    ).fetchall()}

    ec_rows = conn.execute(
        "SELECT email_id, cluster_id FROM email_clusters WHERE cluster_id IS NOT NULL"
    ).fetchall()
    cluster_members: dict[int, list[int]] = {}
    for eid, cid in ec_rows:
        cluster_members.setdefault(cid, []).append(eid)

    print(f"  {len(ec_rows):,} email-cluster vazby, nacitavam embeddingy...", flush=True)
    all_eids = list({r[0] for r in ec_rows})
    emb_map: dict[int, np.ndarray] = {}
    CHUNK = 5000
    for i in range(0, len(all_eids), CHUNK):
        chunk = all_eids[i : i + CHUNK]
        ph = ",".join("?" * len(chunk))
        for eid, blob in conn.execute(
            f"SELECT id, embedding FROM emails WHERE id IN ({ph}) AND embedding IS NOT NULL",
            chunk,
        ).fetchall():
            emb_map[eid] = np.frombuffer(blob, dtype=np.float32).copy()

    print(f"  Embeddingy pre centroidy: {len(emb_map):,}", flush=True)

    cluster_ids = sorted(cluster_members.keys())
    dim = next(iter(emb_map.values())).shape[0] if emb_map else 768
    mat = np.zeros((len(cluster_ids), dim), dtype=np.float32)
    for i, cid in enumerate(cluster_ids):
        vecs = [emb_map[e] for e in cluster_members[cid] if e in emb_map]
        if vecs:
            c = np.mean(vecs, axis=0).astype(np.float32)
            n = np.linalg.norm(c)
            mat[i] = c / n if n > 0 else c

    print(f"  Centroidov: {len(cluster_ids)}", flush=True)
    return mat, cluster_ids, labels


def _load_all_embeddings(conn):
    """Return list of (email_id, normalised_vec)."""
    print("  Nacitavam vsetky embeddingy mailov...", flush=True)
    rows = conn.execute(
        "SELECT id, embedding FROM emails WHERE embedding IS NOT NULL"
    ).fetchall()
    result = []
    for eid, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        n = np.linalg.norm(vec)
        result.append((eid, vec / n if n > 0 else vec))
    print(f"  Nacitanych: {len(result):,}", flush=True)
    return result


def _topics_for_batch(vecs_batch: np.ndarray, centroid_mat: np.ndarray,
                      cluster_ids: list[int]) -> list[list[tuple]]:
    """Return per-email list of (cluster_id, sim, rank, low_confidence)."""
    sims_mat = vecs_batch @ centroid_mat.T          # (batch, n_clusters)
    results = []
    for sims in sims_mat:
        threshold = sims.mean() + 2.0 * sims.std()
        sel = np.where(sims >= threshold)[0]
        low_conf = 0
        if len(sel) == 0:
            sel = np.array([int(np.argmax(sims))])
            low_conf = 1
        sel = sel[np.argsort(sims[sel])[::-1]]     # sort descending
        results.append([
            (cluster_ids[idx], float(sims[idx]), rank + 1, low_conf)
            for rank, idx in enumerate(sel)
        ])
    return results


def main():
    _fix_stdout()

    conn = sqlite3.connect(DB_PATH)

    print("=== email_topics: vytvoram tabulku ===", flush=True)
    _ensure_table(conn)
    existing = conn.execute("SELECT COUNT(*) FROM email_topics").fetchone()[0]
    if existing:
        print(f"  Existuje {existing:,} riadkov — mazem a prepocitavam...", flush=True)
        conn.execute("DELETE FROM email_topics")
        conn.commit()

    print("\n=== Centroidy ===", flush=True)
    centroid_mat, cluster_ids, labels = _load_centroids(conn)

    print("\n=== Embeddingy mailov ===", flush=True)
    all_emails = _load_all_embeddings(conn)

    print(f"\n=== Pocitam temy pre {len(all_emails):,} mailov ===", flush=True)

    BATCH = 2000
    t0 = time.time()
    total_rows = 0

    for start in range(0, len(all_emails), BATCH):
        batch = all_emails[start : start + BATCH]
        eids   = [e[0] for e in batch]
        vecs   = np.array([e[1] for e in batch], dtype=np.float32)

        batch_topics = _topics_for_batch(vecs, centroid_mat, cluster_ids)

        rows = []
        for eid, topics in zip(eids, batch_topics):
            rows.extend((eid, cid, sim, rank, lc) for cid, sim, rank, lc in topics)

        conn.executemany(
            "INSERT OR REPLACE INTO email_topics "
            "(email_id, cluster_id, similarity, rank, low_confidence) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
        total_rows += len(rows)

        done = start + len(batch)
        if done % 10000 == 0 or done == len(all_emails):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(all_emails) - done) if done < len(all_emails) else 0
            print(f"  [{done:,}/{len(all_emails):,}]  {elapsed:.0f}s  ETA {eta:.0f}s  "
                  f"rows={total_rows:,}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Hotovo: {total_rows:,} riadkov v {elapsed:.1f}s", flush=True)

    # ── distribúcia ───────────────────────────────────────────────────────────
    print("\n=== DISTRIBUCIA poctu tem na mail ===", flush=True)
    dist_rows = conn.execute("""
        SELECT topic_count, COUNT(*) AS n
        FROM (SELECT email_id, COUNT(*) AS topic_count FROM email_topics GROUP BY email_id)
        GROUP BY topic_count ORDER BY topic_count
    """).fetchall()

    total_mails = sum(r[1] for r in dist_rows)
    for tc, mc in dist_rows:
        bar = "#" * max(1, mc * 40 // total_mails)
        print(f"  {tc:>2} tem: {mc:>7,}  ({mc/total_mails*100:5.1f}%)  {bar}")

    low_conf_n = conn.execute(
        "SELECT COUNT(DISTINCT email_id) FROM email_topics WHERE low_confidence=1"
    ).fetchone()[0]
    print(f"\n  Low-confidence (fallback top-1): {low_conf_n:,}  ({low_conf_n/total_mails*100:.1f}%)")

    # ── testovaci profily ─────────────────────────────────────────────────────
    print("\n=== PROFILY testovacich mailov ===", flush=True)

    test_queries = [
        ("Patronka/2202",
         "SELECT id, subject, from_address FROM emails "
         "WHERE (LOWER(subject) LIKE '%patronka%' OR subject LIKE '%2202%') "
         "AND embedding IS NOT NULL LIMIT 2"),
        ("Eurovea/Tower",
         "SELECT id, subject, from_address FROM emails "
         "WHERE (LOWER(subject) LIKE '%eurovea%' OR LOWER(subject) LIKE '%tower%') "
         "AND embedding IS NOT NULL LIMIT 2"),
        ("Svetlotechnika",
         "SELECT id, subject, from_address FROM emails "
         "WHERE LOWER(subject) LIKE '%svetlotechn%' "
         "AND embedding IS NOT NULL LIMIT 2"),
        ("Nahodny",
         "SELECT id, subject, from_address FROM emails "
         "WHERE embedding IS NOT NULL AND body_text IS NOT NULL AND body_text != '' "
         "ORDER BY RANDOM() LIMIT 2"),
    ]

    for group, sql in test_queries:
        for eid, subj, frm in conn.execute(sql).fetchall():
            topics = conn.execute("""
                SELECT et.rank, et.similarity, et.low_confidence, c.label
                FROM email_topics et JOIN clusters c ON et.cluster_id = c.id
                WHERE et.email_id = ? ORDER BY et.rank
            """, (eid,)).fetchall()

            print(f"\n  [{group}]  id={eid}")
            print(f"  Subject : {(subj or '')[:70]}")
            print(f"  From    : {(frm or '')[:50]}")
            lc = " [LOW_CONF]" if topics and topics[0][2] else ""
            print(f"  Temy{lc}:")
            for rank, sim, _, label in topics:
                print(f"    #{rank}  {sim:.3f}  {(label or 'n/a')[:65]}")
            if not topics:
                print("    (ziadne temy — email nema embedding?)")

    conn.close()
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
