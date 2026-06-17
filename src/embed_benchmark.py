"""Benchmark 3 embedding models on ~300 representative emails.

Measures:
  A) Similarity distribution (min/max/mean/std/percentiles)
  B) Separation gap = mean(related pairs) - mean(unrelated pairs)

Usage: python -m src.embed_benchmark
"""

import os
import sqlite3
import sys
import time
from itertools import combinations

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH     = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_URL  = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELS      = ["nomic-embed-text", "bge-m3", "qwen3-embedding:4b"]

# ── sample selection ──────────────────────────────────────────────────────────

GROUPS = {
    "patronka":      ("(LOWER(subject) LIKE '%patronka%' OR subject LIKE '%2202%')", 45),
    "eurovea_tower": ("(LOWER(subject) LIKE '%eurovea%' OR LOWER(subject) LIKE '%tower%')", 45),
    "westend":       ("LOWER(subject) LIKE '%westend%'", 40),
    "svetlotechnika":("LOWER(subject) LIKE '%svetlotechn%'", 40),
    "faktura":       ("(LOWER(subject) LIKE '%faktúr%' OR LOWER(subject) LIKE '%faktura%' OR LOWER(subject) LIKE '%invoice%')", 35),
    "newsletter":    ("(LOWER(from_address) LIKE '%no-reply%' OR LOWER(from_address) LIKE '%noreply%' OR LOWER(from_address) LIKE '%newsletter%')", 35),
    "klub_social":   ("(LOWER(subject) LIKE '%volejbal%' OR LOWER(subject) LIKE '%večer%' OR LOWER(subject) LIKE '%vecer%' OR LOWER(subject) LIKE '%klub%' OR LOWER(subject) LIKE '%dovolenk%')", 35),
}

# Related group pairs (same group) — measured within-group
RELATED_GROUPS = [
    ("patronka",       "patronka"),
    ("eurovea_tower",  "eurovea_tower"),
    ("westend",        "westend"),
    ("svetlotechnika", "svetlotechnika"),
]

# Unrelated cross-group pairs
UNRELATED_GROUPS = [
    ("patronka",       "faktura"),
    ("patronka",       "newsletter"),
    ("svetlotechnika", "klub_social"),
    ("eurovea_tower",  "faktura"),
    ("westend",        "newsletter"),
]

MAX_PAIRS = 500  # cap per group-pair to keep it fast


def select_sample(conn) -> dict[str, list[dict]]:
    sample: dict[str, list[dict]] = {}
    for grp, (cond, limit) in GROUPS.items():
        rows = conn.execute(
            f"SELECT id, subject, from_address, body_text "
            f"FROM emails WHERE {cond} AND body_text IS NOT NULL AND body_text != '' "
            f"ORDER BY RANDOM() LIMIT {limit}"
        ).fetchall()
        sample[grp] = [{"id": r[0], "subject": r[1] or "", "from": r[2] or "",
                        "text": r[3] or ""} for r in rows]
        print(f"  {grp:<18} {len(sample[grp]):>3} emailov")
    return sample


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_text(text: str, model: str) -> np.ndarray | None:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": model, "prompt": text[:1500]},
            timeout=120,
        )
        r.raise_for_status()
        vec = np.array(r.json()["embedding"], dtype=np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    except Exception as e:
        print(f"    ERR {model}: {e}", file=sys.stderr)
        return None


def embed_group(sample: dict[str, list[dict]], model: str) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {g: [] for g in sample}
    total = sum(len(v) for v in sample.values())
    done = 0
    t0 = time.time()
    for grp, emails in sample.items():
        for em in emails:
            input_text = f"{em['subject']} {em['text'][:500]}"
            vec = embed_text(input_text, model)
            if vec is not None:
                result[grp].append(vec)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"    [{done}/{total}]  {elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)
    elapsed = time.time() - t0
    print(f"    Hotovo: {done} emailov v {elapsed:.1f}s  ({elapsed/done:.3f}s/email)")
    return result


# ── similarity metrics ────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # already normalized


def pairwise_sims(vecs_a: list[np.ndarray], vecs_b: list[np.ndarray],
                  same_group: bool = False, max_pairs: int = MAX_PAIRS) -> list[float]:
    sims = []
    if same_group:
        pairs = list(combinations(range(len(vecs_a)), 2))
    else:
        pairs = [(i, j) for i in range(len(vecs_a)) for j in range(len(vecs_b))]
    # cap
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    for i, j in pairs:
        sims.append(cosine(vecs_a[i], vecs_b[j]))
    return sims


def distribution_stats(sims: list[float]) -> dict:
    if not sims:
        return {}
    arr = np.array(sims)
    return {
        "n":    len(arr),
        "min":  float(arr.min()),
        "p10":  float(np.percentile(arr, 10)),
        "mean": float(arr.mean()),
        "p90":  float(np.percentile(arr, 90)),
        "max":  float(arr.max()),
        "std":  float(arr.std()),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=== Výber vzorky ===")
    sample = select_sample(conn)
    total = sum(len(v) for v in sample.values())
    print(f"  SPOLU: {total} emailov\n")
    conn.close()

    results = {}

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model}")
        print(f"{'='*60}")

        # Check model available
        try:
            r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                              json={"model": model, "prompt": "test"}, timeout=30)
            r.raise_for_status()
            dim = len(r.json()["embedding"])
            print(f"  Dimenzia: {dim}")
        except Exception as e:
            print(f"  SKIP — model nedostupný: {e}")
            continue

        t_start = time.time()
        embeddings = embed_group(sample, model)
        t_total = time.time() - t_start
        results[model] = {"embeddings": embeddings, "time": t_total, "dim": dim}

    print(f"\n\n{'='*70}")
    print("  VÝSLEDKY")
    print(f"{'='*70}\n")

    # header
    hdr = f"  {'Model':<25} {'Dim':>5}  {'Čas':>6}  {'Min':>6} {'Mean':>6} {'Max':>6} {'Std':>6}  "
    hdr += f"{'RelM':>6}  {'UnrelM':>6}  {'GAP':>6}"
    print(hdr)
    print("  " + "-" * 90)

    for model, res in results.items():
        emb = res["embeddings"]
        dim = res["dim"]
        t   = res["time"]

        # A) overall distribution — all within-group pairs combined
        all_sims: list[float] = []
        for g, vecs in emb.items():
            if len(vecs) >= 2:
                all_sims.extend(pairwise_sims(vecs, vecs, same_group=True, max_pairs=200))
        dst = distribution_stats(all_sims)

        # B) related vs unrelated
        rel_sims: list[float] = []
        for ga, gb in RELATED_GROUPS:
            va, vb = emb.get(ga, []), emb.get(gb, [])
            if va and vb:
                rel_sims.extend(pairwise_sims(va, vb, same_group=(ga == gb), max_pairs=200))

        unrel_sims: list[float] = []
        for ga, gb in UNRELATED_GROUPS:
            va, vb = emb.get(ga, []), emb.get(gb, [])
            if va and vb:
                unrel_sims.extend(pairwise_sims(va, vb, same_group=False, max_pairs=200))

        rel_mean   = float(np.mean(rel_sims))   if rel_sims   else 0.0
        unrel_mean = float(np.mean(unrel_sims)) if unrel_sims else 0.0
        gap        = rel_mean - unrel_mean

        row = (f"  {model:<25} {dim:>5}  {t:>5.0f}s"
               f"  {dst.get('min',0):>6.3f} {dst.get('mean',0):>6.3f} {dst.get('max',0):>6.3f}"
               f" {dst.get('std',0):>6.3f}"
               f"  {rel_mean:>6.3f}  {unrel_mean:>6.3f}  {gap:>6.3f}")
        print(row)

    print()
    print("  Stĺpce: RelM=mean(súvisiace)  UnrelM=mean(nesúvisiace)  GAP=RelM-UnrelM")
    print()

    # Detailed breakdown per related/unrelated group pair
    print(f"\n  DETAIL — related páry:")
    for model, res in results.items():
        print(f"    {model}:")
        emb = res["embeddings"]
        for ga, gb in RELATED_GROUPS:
            va, vb = emb.get(ga, []), emb.get(gb, [])
            if va and vb:
                sims = pairwise_sims(va, vb, same_group=(ga == gb), max_pairs=300)
                print(f"      {ga+'<->'+gb:<35} n={len(sims):>4}  mean={np.mean(sims):.3f}  std={np.std(sims):.3f}")

    print(f"\n  DETAIL — unrelated páry:")
    for model, res in results.items():
        print(f"    {model}:")
        emb = res["embeddings"]
        for ga, gb in UNRELATED_GROUPS:
            va, vb = emb.get(ga, []), emb.get(gb, [])
            if va and vb:
                sims = pairwise_sims(va, vb, same_group=False, max_pairs=300)
                print(f"      {ga+'<->'+gb:<35} n={len(sims):>4}  mean={np.mean(sims):.3f}  std={np.std(sims):.3f}")


if __name__ == "__main__":
    main()
