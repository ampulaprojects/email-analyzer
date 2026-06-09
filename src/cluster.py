"""HDBSCAN clustering over email embeddings with UMAP dimensionality reduction."""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv

try:
    from .db import get_connection
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.db import get_connection

load_dotenv()

DB_PATH = os.environ.get("DB_PATH", "data/emails.db")

UMAP_COMPONENTS  = 50
UMAP_NEIGHBORS   = 15
UMAP_MIN_DIST    = 0.0
UMAP_METRIC      = "cosine"

HDBSCAN_MIN_CLUSTER  = 15
HDBSCAN_MIN_SAMPLES  = 5
HDBSCAN_METRIC       = "euclidean"

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_embeddings(conn) -> tuple[np.ndarray, list[int]]:
    """Return (matrix N x 768, list of email_ids) for all rows with embeddings."""
    rows = conn.execute(
        "SELECT id, embedding FROM emails WHERE embedding IS NOT NULL ORDER BY id ASC"
    ).fetchall()
    if not rows:
        return np.empty((0, 768), dtype=np.float32), []

    email_ids = [r["id"] for r in rows]
    matrix = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    return matrix, email_ids


def _clear_previous(conn) -> None:
    conn.execute("DELETE FROM email_clusters WHERE source = 'hdbscan'")
    conn.execute("DELETE FROM clusters")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'clusters'")
    conn.commit()


def _save_results(
    conn,
    email_ids: list[int],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    now = datetime.now(timezone.utc).isoformat()

    # build cluster map: hdbscan_label → db_cluster_id
    unique_labels = sorted(set(labels.tolist()))
    actual_labels = [l for l in unique_labels if l >= 0]

    cluster_id_map: dict[int, int] = {}
    for hdb_label in actual_labels:
        mask = labels == hdb_label
        size = int(mask.sum())
        cur = conn.execute(
            "INSERT INTO clusters (label, size, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (f"cluster_{hdb_label}", size, now, now),
        )
        cluster_id_map[hdb_label] = cur.lastrowid

    # insert email_clusters
    batch = []
    for email_id, label, prob in zip(email_ids, labels.tolist(), probabilities.tolist()):
        db_cluster_id = cluster_id_map.get(label)  # None for noise (label == -1)
        batch.append((email_id, db_cluster_id, float(prob), "hdbscan", now))

    conn.executemany(
        "INSERT INTO email_clusters (email_id, cluster_id, confidence, source, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        batch,
    )
    conn.commit()


# ── clustering pipeline ───────────────────────────────────────────────────────

def run_clustering(
    db_path: str,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    umap_components: int = UMAP_COMPONENTS,
) -> dict:
    try:
        import umap
    except ImportError:
        raise ImportError("Nainštaluj: pip install umap-learn")
    try:
        import hdbscan
    except ImportError:
        raise ImportError("Nainštaluj: pip install hdbscan")

    conn = get_connection(db_path)

    # ── 1. load embeddings ────────────────────────────────────────────────────
    log.info("Nacitavam embeddingy z DB...")
    t0 = time.time()
    matrix, email_ids = _load_embeddings(conn)
    n = len(email_ids)
    if n == 0:
        log.info("Ziadne embeddingy v DB — spusti najprv embed.py")
        conn.close()
        return {"error": "no_embeddings"}

    log.info("Nacitanych %d embeddingov (%.1f s)", n, time.time() - t0)

    # ── 2. UMAP ───────────────────────────────────────────────────────────────
    n_components = min(umap_components, n - 2)  # UMAP needs n > n_components
    log.info(
        "UMAP: %d x %d -> %d dimenzii  (neighbors=%d, metric=%s)...",
        n, matrix.shape[1], n_components, UMAP_NEIGHBORS, UMAP_METRIC,
    )
    t1 = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
        verbose=False,
    )
    reduced = reducer.fit_transform(matrix)
    log.info("UMAP hotovy (%.1f s)  shape %s", time.time() - t1, reduced.shape)

    # ── 3. HDBSCAN ────────────────────────────────────────────────────────────
    log.info(
        "HDBSCAN: min_cluster_size=%d, min_samples=%d, metric=%s...",
        min_cluster_size, min_samples, HDBSCAN_METRIC,
    )
    t2 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=HDBSCAN_METRIC,
        prediction_data=True,
    )
    clusterer.fit(reduced)
    labels: np.ndarray = clusterer.labels_
    probs:  np.ndarray = clusterer.probabilities_
    log.info("HDBSCAN hotovy (%.1f s)", time.time() - t2)

    # ── 4. save results ───────────────────────────────────────────────────────
    log.info("Ukladam vysledky do DB...")
    _clear_previous(conn)
    _save_results(conn, email_ids, labels, probs)
    conn.close()

    # ── 5. stats ──────────────────────────────────────────────────────────────
    n_noise    = int((labels == -1).sum())
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    top10 = sorted(zip(counts.tolist(), unique.tolist()), reverse=True)[:10]

    return {
        "total":      n,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "noise_pct":  round(n_noise / n * 100, 1),
        "top10":      top10,  # [(size, hdbscan_label), ...]
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HDBSCAN clustering nad embeddings")
    parser.add_argument("--min-cluster-size", type=int, default=HDBSCAN_MIN_CLUSTER,
                        help=f"HDBSCAN min_cluster_size (default: {HDBSCAN_MIN_CLUSTER})")
    parser.add_argument("--min-samples",      type=int, default=HDBSCAN_MIN_SAMPLES,
                        help=f"HDBSCAN min_samples (default: {HDBSCAN_MIN_SAMPLES})")
    parser.add_argument("--umap-components",  type=int, default=UMAP_COMPONENTS,
                        help=f"UMAP output dimensions (default: {UMAP_COMPONENTS})")
    args = parser.parse_args()

    t_start = time.time()
    result = run_clustering(
        DB_PATH,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        umap_components=args.umap_components,
    )

    if "error" in result:
        sys.exit(1)

    elapsed = time.time() - t_start

    print("\n--- Clustering summary ----------------------------------------")
    print(f"  Emailov so embeddingy : {result['total']:>6}")
    print(f"  Pocet zhlukov         : {result['n_clusters']:>6}")
    print(f"  Noise (cluster -1)    : {result['n_noise']:>6}  ({result['noise_pct']:.1f}%)")
    print(f"  Celkovy cas           : {elapsed:>6.1f} s")
    print()
    print("  TOP 10 zhlukov (velkost):")
    for rank, (size, label) in enumerate(result["top10"], 1):
        bar = "#" * min(size // max(result["top10"][0][0] // 30, 1), 40)
        print(f"    {rank:2}. cluster_{label:<4}  {size:>5} emailov  {bar}")
    print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
