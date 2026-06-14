"""Person analysis: contacts and roles extracted from a set of emails."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

try:
    from .db import get_connection
    from .search import search as email_search
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.db import get_connection
    from src.search import search as email_search

load_dotenv()

DB_PATH         = os.environ.get("DB_PATH", "data/emails.db")
INTERNAL_DOMAIN = "gfi.sk"
TOP_N_MAIN      = 3
ACTIVE_DAYS     = 60


# ── helpers ───────────────────────────────────────────────────────────────────

def _domain(addr: str) -> str:
    return addr.split("@", 1)[1].lower().strip() if "@" in addr else ""


def _parse_addrs(json_str) -> list[str]:
    if not json_str:
        return []
    try:
        return [
            a.get("address", "").lower().strip()
            for a in json.loads(json_str)
            if a.get("address", "").strip()
        ]
    except Exception:
        return []


# ── core analysis ─────────────────────────────────────────────────────────────

def analyze_persons(email_ids: list[int], db_path: str = DB_PATH) -> list[dict]:
    """Return person stats for a set of email IDs, sorted by email_count desc."""
    if not email_ids:
        return []

    conn = get_connection(db_path)
    ph   = ",".join("?" * len(email_ids))
    rows = conn.execute(
        f"""
        SELECT id, from_address, to_addresses, cc_addresses, date
        FROM emails WHERE id IN ({ph}) ORDER BY date
        """,
        email_ids,
    ).fetchall()
    conn.close()

    persons: dict[str, dict] = {}

    def _get(addr: str) -> dict:
        if addr not in persons:
            dom = _domain(addr)
            persons[addr] = dict(
                address      = addr,
                domain       = dom,
                is_internal  = (dom == INTERNAL_DOMAIN),
                email_count  = 0,
                as_sender    = 0,
                as_recipient = 0,
                first_date   = None,
                last_date    = None,
            )
        return persons[addr]

    def _upd_dates(p: dict, d: str | None) -> None:
        if not d:
            return
        d = d[:10]
        if p["first_date"] is None or d < p["first_date"]:
            p["first_date"] = d
        if p["last_date"]  is None or d > p["last_date"]:
            p["last_date"]  = d

    for row in rows:
        sender = (row["from_address"] or "").lower().strip()
        date   = row["date"]
        recips = _parse_addrs(row["to_addresses"]) + _parse_addrs(row["cc_addresses"])

        if sender:
            p = _get(sender)
            p["email_count"] += 1
            p["as_sender"]   += 1
            _upd_dates(p, date)

        seen = {sender}
        for addr in recips:
            if not addr or addr in seen:
                continue
            seen.add(addr)
            p = _get(addr)
            p["email_count"]  += 1
            p["as_recipient"] += 1
            _upd_dates(p, date)

    today  = datetime.now(timezone.utc).date().isoformat()
    result = sorted(persons.values(), key=lambda x: x["email_count"], reverse=True)

    for rank, p in enumerate(result):
        labels = ["interny GFI" if p["is_internal"] else f"externi — {p['domain']}"]
        if rank < TOP_N_MAIN:
            labels.append("hlavny kontakt")

        last = p["last_date"]
        if last:
            days_ago    = (datetime.fromisoformat(today) - datetime.fromisoformat(last)).days
            p["active"] = days_ago <= ACTIVE_DAYS
        else:
            p["active"] = False

        p["role"] = ", ".join(labels)

    return result


# ── display ───────────────────────────────────────────────────────────────────

_FMT = "  {:<36} {:<4} {:>5}  {:<12}  {:<7}  {:<7}  {:<32}  {}"


def _hdr() -> None:
    print(_FMT.format("Adresa", "I/E", "Em", "Smer S/R", "Od", "Do", "Rola", "Aktiv?"))
    print("  " + "-" * 115)


def _row(p: dict) -> None:
    smer   = f"S:{p['as_sender']:>3}  R:{p['as_recipient']:>3}"
    od     = (p["first_date"] or "")[:7]
    do_    = (p["last_date"]  or "")[:7]
    ie     = "INT" if p["is_internal"] else "EXT"
    active = "ANO" if p["active"] else "nie"
    print(_FMT.format(p["address"][:36], ie, p["email_count"], smer, od, do_, p["role"][:32], active))


def print_persons(persons: list[dict], title: str) -> None:
    internal = [p for p in persons if p["is_internal"]]
    external = [p for p in persons if not p["is_internal"]]

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  Osob celkom: {len(persons)}  |  internych: {len(internal)}  |  externych: {len(external)}")
    print(f"{'=' * 70}")

    if internal:
        print(f"\n  -- INTERNI GFI ({len(internal)}) --")
        _hdr()
        for p in internal:
            _row(p)

    if external:
        print(f"\n  -- EXTERNI ({len(external)}) --")
        _hdr()
        for p in external:
            _row(p)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze persons from email search results")
    parser.add_argument("query",
                        help="Search query (project name, topic, ...)")
    parser.add_argument("--top",       type=int,   default=50,   help="Max search results (default 50)")
    parser.add_argument("--min-score", type=float, default=0.50, help="Min search score (default 0.50)")
    parser.add_argument("--db",        default=DB_PATH,          help="Path to SQLite DB")
    parser.add_argument("--limit",     type=int,   default=None, help="Show only top N persons")
    args = parser.parse_args()

    print(f'\nQuery: "{args.query}"  (top={args.top}, min_score={args.min_score})')

    results    = email_search(args.query, db_path=args.db, top_k=args.top, min_score=args.min_score)
    email_ids  = [r["id"] for r in results]
    n_direct   = sum(1 for r in results if not r.get("expanded"))
    n_expanded = sum(1 for r in results if r.get("expanded"))
    print(f"Emailov: {len(email_ids)}  (direct: {n_direct}, thread+: {n_expanded})")

    persons = analyze_persons(email_ids, db_path=args.db)
    if args.limit:
        persons = persons[:args.limit]

    print_persons(persons, f'Osoby: "{args.query}"')


if __name__ == "__main__":
    main()
