"""Query over precomputed community graph (data/communities.json)."""

import argparse
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

try:
    from .search  import search as email_search
    from .persons import analyze_persons
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.search  import search as email_search
    from src.persons import analyze_persons

DB_PATH          = os.environ.get("DB_PATH",          "data/emails.db")
COMMUNITIES_PATH = os.environ.get("COMMUNITIES_PATH", "data/communities.json")
INTERNAL_DOMAIN  = "gfi.sk"
ACTIVE_DAYS      = 60


def _is_internal(addr: str) -> bool:
    return addr.endswith("@" + INTERNAL_DOMAIN)


def _load_communities(path: str) -> tuple[list[dict], dict[str, int]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["communities"], data["person_to_community"]


def ask(
    text: str,
    db_path:          str   = DB_PATH,
    communities_path: str   = COMMUNITIES_PATH,
    top_k:            int   = 50,
    min_score:        float = 0.50,
) -> dict | None:
    """
    1. search(text) → email_ids
    2. analyze_persons(email_ids) → persons in results
    3. map persons → communities via communities.json (no graph recompute)
    4. return structured result for dominant community
    """
    # Step 1: search
    results   = email_search(text, db_path=db_path, top_k=top_k, min_score=min_score)
    email_ids = [r["id"] for r in results]
    n_direct  = sum(
        1 for r in results
        if not r.get("expanded") and not r.get("person_expanded")
    )

    if not email_ids:
        return None

    # Step 2: persons from search results
    persons_list = analyze_persons(email_ids, db_path=db_path)

    # Step 3: load communities; tally votes weighted by email_count
    communities, p2c = _load_communities(communities_path)

    com_score:   dict[int, int]        = {}
    com_persons: dict[int, list[dict]] = {}

    for p in persons_list:
        cid = p2c.get(p["address"])
        if cid is None:
            continue
        com_score[cid]    = com_score.get(cid, 0) + p["email_count"]
        com_persons.setdefault(cid, []).append(p)

    if not com_score:
        return None

    dominant_cid = max(com_score, key=com_score.get)
    community    = communities[dominant_cid]

    # Step 4: is the community still active?
    today     = datetime.now(timezone.utc).date().isoformat()
    last_date = community.get("last", "")
    active    = False
    if last_date:
        try:
            days_since = (
                datetime.fromisoformat(today) -
                datetime.fromisoformat(last_date + "-01")
            ).days
            active = days_since <= ACTIVE_DAYS
        except ValueError:
            pass

    return {
        "query":           text,
        "email_ids":       email_ids,
        "n_direct":        n_direct,
        "community_id":    dominant_cid,
        "com_score":       com_score,
        "hit_score":       com_score[dominant_cid],
        "matched_persons": com_persons.get(dominant_cid, []),
        "community":       community,
        "all_communities": communities,
        "active":          active,
    }


def print_result(r: dict | None, query: str = "") -> None:
    if r is None:
        print(f"[Ziadne vysledky pre: {query!r}]")
        return

    c      = r["community"]
    active = "AKTIVNA" if r["active"] else "neaktivna"

    print(f"\n{'=' * 68}")
    print(f'  DOTAZ: "{r["query"]}"')
    print(
        f"  Komunita #{r['community_id']:>2}  |  "
        f"{c['size']} osob ({c['n_int']} INT / {c['n_ext']} EXT)  |  "
        f"skore={r['hit_score']}"
    )
    print(f"{'=' * 68}")

    # ── IDENTITA ─────────────────────────────────────────────────────────────
    words = [w for w, _ in c["top_words"][:8]]
    print(f"\nIDENTITA:")
    print(f"  Temy    : {' · '.join(words)}")
    if c["ext_doms"]:
        doms = "  ".join(f"{d}({n})" for d, n in c["ext_doms"])
        print(f"  Partneri: {doms}")

    # ── ALTERNATÍVNE MENÁ ────────────────────────────────────────────────────
    # words appearing in > 1/40 of community emails → candidate project identifiers
    threshold = max(5, c["n_emails"] // 40)
    alt_names = [w for w, n in c["top_words"] if n >= threshold]
    print(f"\nALTERNATIVNE MENA / JADRO KOMUNITY:")
    print(f"  {' · '.join(alt_names)}")

    # ── ĽUDIA ────────────────────────────────────────────────────────────────
    matched  = sorted(r["matched_persons"], key=lambda p: p["email_count"], reverse=True)
    top5_set = set(c.get("top5", []))

    print(f"\nLUDIA (z vysledkov dotazu — komunita #{r['community_id']}):")
    if matched:
        for p in matched[:12]:
            ie   = "INT" if _is_internal(p["address"]) else "EXT"
            smer = f"S:{p['as_sender']:>3} R:{p['as_recipient']:>3}"
            tags = []
            if p["address"] in top5_set:
                tags.append("top-centralita")
            if p.get("active"):
                tags.append("aktivny")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            print(f"  [{ie}]  {p['address']:<42} {smer}  {p['role'][:32]}{tag_str}")
    else:
        print("  (ziadne osoby z dotazu sa nenasli v tejto komunite)")

    print(f"\n  Top 5 grafu (centralita) — komunita #{r['community_id']}:")
    for p in c.get("top5", []):
        ie    = "INT" if _is_internal(p) else "EXT"
        match = " <- v dotaze" if any(x["address"] == p for x in r["matched_persons"]) else ""
        print(f"    [{ie}]  {p}{match}")

    # ── ROZSAH ───────────────────────────────────────────────────────────────
    print(f"\nROZSAH:")
    print(
        f"  {c['first']} – {c['last']}  |  "
        f"{c['n_emails']} emailov v komunite  |  {active}"
    )
    print(
        f"  Vysledky dotazu: {len(r['email_ids'])} emailov  "
        f"({r['n_direct']} priamych)"
    )

    # ── ostatné komunity s hitmi ──────────────────────────────────────────────
    others = sorted(
        ((cid, score) for cid, score in r["com_score"].items()
         if cid != r["community_id"]),
        key=lambda x: x[1], reverse=True,
    )[:3]
    if others:
        all_coms = r["all_communities"]
        print(f"\n  Dalsie komunity s hitmi:")
        for cid, score in others:
            oc   = all_coms[cid] if cid < len(all_coms) else {}
            ow   = [w for w, _ in oc.get("top_words", [])[:4]]
            print(f"    #{cid:>2}  skore={score:>4}  temy={ow}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query over precomputed community graph")
    parser.add_argument("query",                              help="Search query text")
    parser.add_argument("--db",        default=DB_PATH,      help="SQLite DB path")
    parser.add_argument("--graph",     default=COMMUNITIES_PATH, help="communities.json path")
    parser.add_argument("--top-k",     type=int,   default=50,   help="Max search results (default 50)")
    parser.add_argument("--min-score", type=float, default=0.50, help="Min search score (default 0.50)")
    args = parser.parse_args()

    result = ask(
        args.query,
        db_path          = args.db,
        communities_path = args.graph,
        top_k            = args.top_k,
        min_score        = args.min_score,
    )
    print_result(result, args.query)


if __name__ == "__main__":
    main()
