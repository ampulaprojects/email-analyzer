"""Snapshot current state to JSON for baseline comparison after dataset expansion."""

import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

try:
    from .db   import get_connection
    from .ask  import ask, _load_communities
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.db  import get_connection
    from src.ask import ask, _load_communities

DB_PATH          = os.environ.get("DB_PATH",          "data/emails.db")
COMMUNITIES_PATH = os.environ.get("COMMUNITIES_PATH", "data/communities.json")
OUTPUT_PATH      = "data/baseline_12k.json"

REFERENCE_QUERIES = ["Eurovea", "Tower 220", "Westend", "Patronka 2202"]


# ── DB stats ──────────────────────────────────────────────────────────────────

def _db_stats(db_path: str) -> dict:
    conn = get_connection(db_path)
    total    = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    with_body = conn.execute(
        "SELECT COUNT(*) FROM emails WHERE body_text IS NOT NULL AND body_text != ''"
    ).fetchone()[0]
    with_emb = conn.execute(
        "SELECT COUNT(*) FROM emails WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    n_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    conn.close()
    return {
        "total_emails":    total,
        "with_body_text":  with_body,
        "with_embedding":  with_emb,
        "total_clusters":  n_clusters,
    }


def _top_clusters(db_path: str, n: int = 10) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT c.id, c.label, c.size
        FROM clusters c
        ORDER BY c.size DESC
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [{"id": r["id"], "label": r["label"] or "", "size": r["size"]} for r in rows]


# ── Communities snapshot ──────────────────────────────────────────────────────

def _communities_snapshot(communities_path: str) -> dict:
    communities, _ = _load_communities(communities_path)
    return {
        "total":        len(communities),
        "largest_size": communities[0]["size"] if communities else 0,
        "list": [
            {
                "id":        c["id"],
                "size":      c["size"],
                "n_int":     c["n_int"],
                "n_ext":     c["n_ext"],
                "top_words": [w for w, _ in c["top_words"][:6]],
                "ext_doms":  [[d, n] for d, n in c["ext_doms"][:3]],
                "first":     c.get("first", ""),
                "last":      c.get("last", ""),
                "n_emails":  c["n_emails"],
            }
            for c in communities
        ],
    }


# ── Reference queries ─────────────────────────────────────────────────────────

def _run_queries(
    queries:          list[str],
    db_path:          str,
    communities_path: str,
) -> dict:
    results = {}
    for q in queries:
        print(f"  '{q}' ...", end=" ", flush=True)
        r = ask(q, db_path=db_path, communities_path=communities_path,
                top_k=50, min_score=0.50)
        if r is None:
            print("no result")
            results[q] = None
            continue

        c          = r["community"]
        threshold  = max(5, c["n_emails"] // 40)
        alt_names  = [w for w, n in c["top_words"] if n >= threshold]
        matched    = sorted(r["matched_persons"], key=lambda p: p["email_count"], reverse=True)
        top5       = [p["address"] for p in matched[:5]]
        top5_cen   = c.get("top5", [])

        results[q] = {
            "community_id":   r["community_id"],
            "hit_score":      r["hit_score"],
            "n_emails_found": len(r["email_ids"]),
            "n_direct":       r["n_direct"],
            "alt_names":      alt_names,
            "top5_by_hits":   top5,
            "top5_centrality": top5_cen,
            "community_size": c["size"],
            "community_top_words": [w for w, _ in c["top_words"][:8]],
        }
        print(f"komunita #{r['community_id']}  score={r['hit_score']}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def take_snapshot(
    db_path:          str = DB_PATH,
    communities_path: str = COMMUNITIES_PATH,
    output_path:      str = OUTPUT_PATH,
) -> dict:
    print("=== Snapshot ===")

    print("DB štatistiky ...", flush=True)
    db   = _db_stats(db_path)
    print(f"  emails={db['total_emails']}  body={db['with_body_text']}"
          f"  emb={db['with_embedding']}  clusters={db['total_clusters']}")

    print("Top 10 clusterov ...", flush=True)
    top_cl = _top_clusters(db_path)
    for c in top_cl:
        print(f"  id={c['id']:>3}  size={c['size']:>4}  {c['label'][:50]}")

    print("Komunity ...", flush=True)
    com = _communities_snapshot(communities_path)
    print(f"  celkom={com['total']}  najväčšia={com['largest_size']}")

    print("Referenčné dotazy ...", flush=True)
    queries = _run_queries(REFERENCE_QUERIES, db_path, communities_path)

    snapshot = {
        "created_at":        datetime.now(timezone.utc).isoformat(),
        "db_path":           db_path,
        "communities_path":  communities_path,
        "db_stats":          db,
        "top_clusters":      top_cl,
        "communities":       com,
        "reference_queries": queries,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"\nUložené: {output_path}")
    return snapshot


def print_summary(s: dict) -> None:
    db  = s["db_stats"]
    com = s["communities"]

    print(f"\n{'=' * 60}")
    print(f"  BASELINE SNAPSHOT — {s['created_at'][:10]}")
    print(f"{'=' * 60}")

    print(f"\nDATABÁZA:")
    print(f"  Emailov celkom   : {db['total_emails']:>6}")
    print(f"  S body_text      : {db['with_body_text']:>6}")
    print(f"  S embedding      : {db['with_embedding']:>6}")
    print(f"  Clusterov        : {db['total_clusters']:>6}")

    print(f"\nTOP 10 CLUSTEROV:")
    for c in s["top_clusters"]:
        print(f"  id={c['id']:>3}  size={c['size']:>4}  {c['label'][:48]}")

    print(f"\nKOMUNITY ({com['total']} komunít, najväčšia {com['largest_size']}):")
    for c in com["list"]:
        words = " · ".join(c["top_words"][:4])
        doms  = "  ".join(f"{d}({n})" for d, n in c["ext_doms"][:2])
        print(
            f"  #{c['id']:>2}  {c['size']:>4} osôb  "
            f"{c['n_emails']:>5} mailov  {words:<40}  {doms}"
        )

    print(f"\nREFERENČNÉ DOTAZY:")
    for q, r in s["reference_queries"].items():
        if r is None:
            print(f"  '{q}': no result")
            continue
        alt  = " · ".join(r["alt_names"][:5])
        top5 = ", ".join(r["top5_by_hits"][:3])
        print(f"\n  Dotaz    : \"{q}\"")
        print(f"  Komunita : #{r['community_id']}  ({r['community_size']} osôb)"
              f"  score={r['hit_score']}  nájdených={r['n_emails_found']}")
        print(f"  Alt. mená: {alt}")
        print(f"  Top ľudia: {top5}")


if __name__ == "__main__":
    s = take_snapshot()
    print_summary(s)
