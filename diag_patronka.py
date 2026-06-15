"""Diagnostics: score breakdown for all Patronka/2202 emails."""

import re
import sqlite3
from collections import defaultdict

import numpy as np
import requests

DB_PATH     = "data/emails.db"
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

WEIGHT_FTS     = 0.3
WEIGHT_VEC     = 0.5
WEIGHT_CLUSTER = 0.2
NOISE_PENALTY  = -0.1
CLUSTER_TOP_K  = 3
MIN_SCORE      = 0.55

QUERY = "Patronka 2202"


def main():
    # ── 1. Query embedding ────────────────────────────────────────────────────
    print(f"Embedding: {QUERY!r} ...", flush=True)
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": QUERY},
        timeout=60,
    )
    resp.raise_for_status()
    arr   = np.array(resp.json()["embedding"], dtype=np.float32)
    norm  = np.linalg.norm(arr)
    q_vec = arr / norm if norm > 0 else arr

    conn             = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ── 2. Target emails ──────────────────────────────────────────────────────
    target_rows = conn.execute("""
        SELECT id, subject, date, from_address
        FROM emails
        WHERE subject  LIKE '%Patronk%' OR subject  LIKE '%2202%'
           OR body_text LIKE '%Patronk%' OR body_text LIKE '%2202%'
        ORDER BY date
    """).fetchall()
    target_ids = {r["id"] for r in target_rows}
    print(f"Cielove emaily: {len(target_ids)}\n")

    # ── 3. FTS scores ─────────────────────────────────────────────────────────
    words   = re.findall(r'\b\w{2,}\b', QUERY.lower())[:12]
    fts_q   = " OR ".join(f'"{w}"' for w in words)

    # First check total FTS hits to know if LIMIT 300 is cutting anything
    total_fts = conn.execute(
        "SELECT COUNT(*) FROM emails_fts WHERE emails_fts MATCH ?", (fts_q,)
    ).fetchone()[0]
    print(f"Celkove FTS zhody pre '{QUERY}': {total_fts}  (limit v search.py: 300)")

    fts_rows = conn.execute(
        "SELECT rowid, rank FROM emails_fts WHERE emails_fts MATCH ? ORDER BY rank LIMIT 300",
        (fts_q,),
    ).fetchall()
    fts_raw = {int(r[0]): r[1] for r in fts_rows}
    if fts_raw:
        best, worst = min(fts_raw.values()), max(fts_raw.values())
        if best == worst:
            rank_scores = {eid: 1.0 for eid in fts_raw}
        else:
            rank_scores = {eid: (rank - worst) / (best - worst) for eid, rank in fts_raw.items()}
        fts_scores = {eid: 0.7 + 0.3 * rank_scores[eid] for eid in rank_scores}
    else:
        fts_scores = {}

    in_fts_top300 = len(target_ids & set(fts_scores))
    print(f"Z 160 cielovych emailov v FTS top-300: {in_fts_top300}\n")

    # ── 4. All embeddings ─────────────────────────────────────────────────────
    print("Nacitavam embeddingy ...", flush=True)
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
    matrix      = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    norms_      = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms_[norms_ == 0] = 1
    norm_matrix = matrix / norms_
    id_to_idx   = {eid: i for i, eid in enumerate(email_ids)}

    # ── 5. Vector scores (ALL emails, not just top-100) ───────────────────────
    all_sims  = norm_matrix @ q_vec
    vec_scores = {email_ids[i]: float(max(0.0, all_sims[i])) for i in range(len(email_ids))}

    # ── 6. Cluster scores (exact _cluster_search logic) ───────────────────────
    members: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        if cid is not None:
            members[cid].append(i)

    cluster_scores: dict[int, float] = {}
    if members:
        cids = list(members.keys())
        centroid_list = []
        for cid in cids:
            c = norm_matrix[members[cid]].mean(axis=0)
            n = np.linalg.norm(c)
            centroid_list.append(c / n if n > 0 else c)
        sims    = np.stack(centroid_list) @ q_vec
        top_idx = np.argsort(sims)[-CLUSTER_TOP_K:][::-1]
        top3_info = []
        for rank, idx in enumerate(top_idx):
            decay = [1.0, 0.7, 0.5][rank]
            raw   = float(max(0.0, sims[idx]))
            score = raw * decay
            top3_info.append((cids[idx], raw, decay, score, len(members[cids[idx]])))
            for i in members[cids[idx]]:
                eid = email_ids[i]
                if eid not in cluster_scores or cluster_scores[eid] < score:
                    cluster_scores[eid] = score

    for i, cid in enumerate(cluster_ids):
        if cid is None:
            eid = email_ids[i]
            if eid not in cluster_scores:
                cluster_scores[eid] = NOISE_PENALTY

    # Top-3 cluster info
    print("Top-3 clustre pre query (podla centroid cosine):")
    cluster_labels = {
        r[0]: r[1] for r in conn.execute("SELECT id, label FROM clusters").fetchall()
    }
    for cid, raw, decay, score, sz in top3_info:
        lbl = cluster_labels.get(cid, f"cluster_{cid}")
        print(f"  cluster_id={cid:>3}  raw={raw:.4f}  decay={decay}  score={score:.4f}"
              f"  size={sz:>4}  label={lbl[:40]}")
    print()

    # ── 7. Per-email breakdown ────────────────────────────────────────────────
    cluster_label_map = {
        r[0]: r[1] for r in conn.execute("""
            SELECT ec.email_id, c.label
            FROM email_clusters ec JOIN clusters c ON ec.cluster_id = c.id
            WHERE ec.source = 'hdbscan'
        """).fetchall()
    }

    fts_vals   = []
    vec_vals   = []
    clu_vals   = []
    final_vals = []
    passed     = 0

    rows_out = []
    for r in target_rows:
        eid   = r["id"]
        fts   = fts_scores.get(eid, 0.0)
        vec   = vec_scores.get(eid, 0.0)
        clu   = cluster_scores.get(eid, 0.0)
        final = WEIGHT_FTS * fts + WEIGHT_VEC * vec + WEIGHT_CLUSTER * clu
        ok    = final >= MIN_SCORE
        if ok:
            passed += 1

        fts_vals.append(fts)
        vec_vals.append(vec)
        clu_vals.append(clu)
        final_vals.append(final)

        label = cluster_label_map.get(eid, "NOISE")
        rows_out.append((r["date"][:10], eid, fts, vec, clu, final, ok, r["subject"] or "", label))

    # Sort by final score desc for readability
    rows_out.sort(key=lambda x: x[5], reverse=True)

    header = f"  {'Datum':10}  {'ID':>6}  {'FTS':>6}  {'VEC':>6}  {'CLU':>7}  {'FINAL':>6}  {'OK':>4}  Subject"
    sep    = "  " + "-" * 115
    print(header)
    print(sep)
    for date, eid, fts, vec, clu, final, ok, subj, label in rows_out:
        mark = "ANO" if ok else "nie"
        print(f"  {date:10}  {eid:>6}  {fts:6.3f}  {vec:6.3f}  {clu:7.4f}  {final:6.3f}  {mark:>4}  {subj[:40]}  [{label[:28]}]")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUHRN  — query: '{QUERY}'  (min_score={MIN_SCORE})")
    print(f"{'='*70}")
    print(f"  Cielovych emailov : {len(target_ids)}")
    print(f"  Zobrazilo by sa   : {passed}  ({100*passed//len(target_ids)} %)")
    print(f"  Nepreslo filter   : {len(target_ids)-passed}")
    print()
    print(f"  {'Signal':14}  {'Medián':>8}  {'Min':>8}  {'Max':>8}  {'Nulove':>8}")
    print(f"  {'-'*55}")
    print(f"  {'FTS  (w=0.3)':14}  {np.median(fts_vals):8.3f}  {min(fts_vals):8.3f}  {max(fts_vals):8.3f}  {sum(1 for v in fts_vals if v==0.0):>8}")
    print(f"  {'VEC  (w=0.5)':14}  {np.median(vec_vals):8.3f}  {min(vec_vals):8.3f}  {max(vec_vals):8.3f}  {'—':>8}")
    print(f"  {'CLU  (w=0.2)':14}  {np.median(clu_vals):8.4f}  {min(clu_vals):8.4f}  {max(clu_vals):8.4f}  {sum(1 for v in clu_vals if v<0):>8}  (<0 = noise)")
    print(f"  {'FINAL':14}  {np.median(final_vals):8.3f}  {min(final_vals):8.3f}  {max(final_vals):8.3f}")

    # ── 9. Failed-email deep dive ─────────────────────────────────────────────
    failed = [(fts_vals[i], vec_vals[i], clu_vals[i], final_vals[i])
              for i in range(len(final_vals)) if not (final_vals[i] >= MIN_SCORE)]
    if failed:
        f_fts = [x[0] for x in failed]
        f_vec = [x[1] for x in failed]
        f_clu = [x[2] for x in failed]
        f_fin = [x[3] for x in failed]
        print(f"\n  Emaily co NEPRESLI ({len(failed)}):")
        print(f"    FTS   medián={np.median(f_fts):.3f}  priemer={np.mean(f_fts):.3f}  nulove={sum(1 for v in f_fts if v==0)}/{len(f_fts)}")
        print(f"    VEC   medián={np.median(f_vec):.3f}  priemer={np.mean(f_vec):.3f}")
        print(f"    CLU   medián={np.median(f_clu):.4f}  priemer={np.mean(f_clu):.4f}  noise={sum(1 for v in f_clu if v<0)}/{len(f_fts)}")
        print(f"    FINAL medián={np.median(f_fin):.3f}  priemer={np.mean(f_fin):.3f}")
        c_fts = WEIGHT_FTS     * np.mean(f_fts)
        c_vec = WEIGHT_VEC     * np.mean(f_vec)
        c_clu = WEIGHT_CLUSTER * np.mean(f_clu)
        print(f"\n    Priemerny prispevok (weighted) k final_score u nepresanych:")
        print(f"      FTS  * 0.3 = {c_fts:.4f}  ({100*c_fts/(c_fts+c_vec+abs(c_clu)) if c_fts+c_vec > 0 else 0:.0f} %)")
        print(f"      VEC  * 0.5 = {c_vec:.4f}  ({100*c_vec/(c_fts+c_vec+abs(c_clu)) if c_fts+c_vec > 0 else 0:.0f} %)")
        print(f"      CLU  * 0.2 = {c_clu:.4f}")

    conn.close()


main()
