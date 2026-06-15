"""IMAP server diagnostic — count messages per folder vs DB, NO downloading."""

import imaplib
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

from dotenv import load_dotenv

load_dotenv()

IMAP_HOST  = os.environ["IMAP_HOST"]
IMAP_PORT  = int(os.environ.get("IMAP_PORT", 993))
IMAP_USER  = os.environ["IMAP_USER"]
IMAP_PASS  = os.environ["IMAP_PASS"]
DB_PATH    = os.environ.get("DB_PATH", "data/emails.db")

# Our original --since cutoff
SINCE_DATE = "2024-06-08"
SINCE_IMAP = "08-Jun-2024"   # IMAP SEARCH format

# Only show folders with at least this many messages on server
MIN_MESSAGES = 50


# ── helpers ───────────────────────────────────────────────────────────────────

def _connect() -> imaplib.IMAP4_SSL:
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    imap.login(IMAP_USER, IMAP_PASS)
    return imap


def _status(imap: imaplib.IMAP4_SSL, folder: str) -> int:
    """Return MESSAGES count for folder via STATUS (no SELECT needed)."""
    try:
        status, data = imap.status(f'"{folder}"', "(MESSAGES)")
        if status != "OK":
            return -1
        # data[0] like b'"INBOX" (MESSAGES 12345)'
        import re
        m = re.search(rb"MESSAGES\s+(\d+)", data[0])
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def _select_readonly(imap: imaplib.IMAP4_SSL, folder: str) -> int:
    """SELECT folder in read-only mode; return EXISTS count (-1 on error)."""
    try:
        status, data = imap.select(f'"{folder}"', readonly=True)
        if status != "OK":
            return -1
        return int(data[0])
    except Exception:
        return -1


def _search_uids(imap: imaplib.IMAP4_SSL, criterion: str) -> list[int]:
    try:
        status, data = imap.uid("SEARCH", None, criterion)
        if status != "OK" or not data or data[0] == b"":
            return []
        return [int(u) for u in data[0].split()]
    except Exception:
        return []


def _fetch_internaldate(imap: imaplib.IMAP4_SSL, uid: int) -> str:
    """Fetch INTERNALDATE for a single UID; return YYYY-MM-DD or ''."""
    try:
        status, data = imap.uid("FETCH", str(uid), "(INTERNALDATE)")
        if status != "OK" or not data or data[0] is None:
            return ""
        import re
        raw = data[0].decode("utf-8", errors="replace") if isinstance(data[0], bytes) else str(data[0])
        # INTERNALDATE "15-Jun-2026 18:00:00 +0200"
        m = re.search(r'INTERNALDATE\s+"([^"]+)"', raw)
        if not m:
            return ""
        return datetime.strptime(m.group(1)[:11], "%d-%b-%Y").strftime("%Y-%m-%d")
    except Exception:
        return ""


def _db_stats(folder: str) -> dict:
    """Count emails in DB for this folder and return date range + max uid."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT COUNT(*)               AS cnt,
               MIN(date)              AS first_date,
               MAX(date)              AS last_date,
               MAX(imap_uid)          AS max_uid
        FROM emails WHERE folder = ?
    """, (folder,)).fetchone()
    conn.close()
    if not row or row["cnt"] == 0:
        return {"count": 0, "first": "", "last": "", "max_uid": 0}
    return {
        "count":   row["cnt"],
        "first":   (row["first_date"] or "")[:10],
        "last":    (row["last_date"]  or "")[:10],
        "max_uid": row["max_uid"] or 0,
    }


def _list_folders(imap: imaplib.IMAP4_SSL) -> list[str]:
    """Return all selectable folder names."""
    import re
    status, lines = imap.list()
    folders = []
    for line in lines:
        if not isinstance(line, bytes):
            continue
        decoded = line.decode("utf-8", errors="replace")
        # skip \Noselect folders
        if r"\Noselect" in decoded:
            continue
        m = re.search(r'"([^"]+)"\s*$', decoded)
        if m:
            folders.append(m.group(1))
        else:
            m = re.search(r'\)\s+\S+\s+(\S+)\s*$', decoded)
            if m:
                folders.append(m.group(1))
    return folders


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Pripájam sa na {IMAP_HOST}:{IMAP_PORT} ...", flush=True)
    imap = _connect()
    print(f"OK — {IMAP_USER}\n")

    # ── 1. Celkové DB čísla ───────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    total_db = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    db_oldest = (conn.execute("SELECT MIN(date) FROM emails").fetchone()[0] or "")[:10]
    db_newest = (conn.execute("SELECT MAX(date) FROM emails").fetchone()[0] or "")[:10]
    conn.close()
    print(f"DB celkom: {total_db} emailov  ({db_oldest} – {db_newest})\n")

    # ── 2. Zisti všetky foldere a ich počty cez STATUS ───────────────────────
    print("Zisťujem počty správ v priečinkoch (STATUS) ...", flush=True)
    all_folders = _list_folders(imap)
    print(f"  Nájdených priečinkov: {len(all_folders)}")

    folder_counts: list[tuple[str, int]] = []
    for f in all_folders:
        n = _status(imap, f)
        if n > 0:
            folder_counts.append((f, n))

    # Sort by count desc, filter to MIN_MESSAGES
    folder_counts.sort(key=lambda x: x[1], reverse=True)
    big_folders = [(f, n) for f, n in folder_counts if n >= MIN_MESSAGES]
    print(f"  Priečinkov s >={MIN_MESSAGES} správ: {len(big_folders)}")
    print(f"  Priečinkov s 1+ správ: {len(folder_counts)}\n")

    # ── 3. Detailná analýza každého relevantného priečinka ────────────────────
    print(f"Detailná analýza (SELECT + SEARCH)...")
    print(f"  Referenčný dátum syncu: {SINCE_DATE}\n")

    results = []
    for folder, total_server in big_folders:
        print(f"  [{folder}]  {total_server} správ na serveri", flush=True)

        # SELECT (read-only) — needed for SEARCH
        exists = _select_readonly(imap, folder)
        if exists < 0:
            print(f"    ! SELECT zlyhalo, preskakujem")
            continue

        # Count before and after our since cutoff
        uids_before = _search_uids(imap, f"BEFORE {SINCE_IMAP}")
        uids_since  = _search_uids(imap, f"SINCE {SINCE_IMAP}")
        n_before = len(uids_before)
        n_since  = len(uids_since)

        # Date range: INTERNALDATE of first and last UID
        all_uids = sorted(uids_before + uids_since)
        date_first = _fetch_internaldate(imap, all_uids[0])  if all_uids else ""
        date_last  = _fetch_internaldate(imap, all_uids[-1]) if all_uids else ""

        # What's new since our last sync (UID > max_uid in DB)
        db = _db_stats(folder)
        uids_new = [u for u in uids_since if u > db["max_uid"]] if db["max_uid"] else uids_since

        results.append({
            "folder":       folder,
            "server_total": total_server,
            "n_before":     n_before,     # older than our --since cutoff
            "n_since":      n_since,      # in our sync window
            "n_new":        len(uids_new),# UID > our last synced UID
            "date_first":   date_first,
            "date_last":    date_last,
            "db_count":     db["count"],
            "db_first":     db["first"],
            "db_last":      db["last"],
            "db_max_uid":   db["max_uid"],
        })

        print(f"    server={total_server}  pred-cutoff={n_before}  od-cutoff={n_since}"
              f"  nové={len(uids_new)}  v-DB={db['count']}")

    imap.logout()

    # ── 4. Výpis tabuľky ──────────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print(f"  IMAP DIAGNOSTIC — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Since cutoff: {SINCE_DATE}  |  DB celkom: {total_db}")
    print(f"{'=' * 110}")

    hdr = (
        f"  {'Priečinok':<22}  {'Server':>7}  {'Rozsah servera':>21}  "
        f"{'pred-cutoff':>12}  {'od-cutoff':>9}  {'nové':>5}  "
        f"{'v DB':>6}  {'DB rozsah':>21}  {'chýba*':>7}"
    )
    print(hdr)
    print("  " + "-" * 106)

    total_server_all = 0
    total_missing    = 0
    total_new        = 0

    for r in results:
        # "chýba*" = emails in our window that we don't have
        # = (server messages since cutoff) - (DB count) + (newly arrived not yet synced)
        missing_old = r["n_before"]                     # definitely not in DB
        missing_new = r["n_new"]                        # arrived after last sync
        chyba       = missing_old + missing_new

        srv_range  = f"{r['date_first']} – {r['date_last']}" if r['date_first'] else "?"
        db_range   = f"{r['db_first']} – {r['db_last']}"     if r['db_first']  else "—"

        print(
            f"  {r['folder']:<22}  {r['server_total']:>7}  {srv_range:>21}  "
            f"{r['n_before']:>12}  {r['n_since']:>9}  {r['n_new']:>5}  "
            f"{r['db_count']:>6}  {db_range:>21}  {chyba:>7}"
        )
        total_server_all += r["server_total"]
        total_missing    += chyba
        total_new        += r["n_new"]

    print("  " + "-" * 106)
    print(
        f"  {'SPOLU':<22}  {total_server_all:>7}  {'':>21}  "
        f"{'':>12}  {'':>9}  {total_new:>5}  "
        f"{total_db:>6}  {'':>21}  {total_missing:>7}"
    )

    print(f"\n  * chýba = (pred cutoff {SINCE_DATE}) + (nové po poslednom syncu)")
    print(f"\nPriečinky s <{MIN_MESSAGES} správami (celkom {len(folder_counts) - len(big_folders)}):")
    small = [(f, n) for f, n in folder_counts if n < MIN_MESSAGES]
    for f, n in sorted(small, key=lambda x: x[1], reverse=True)[:20]:
        db = _db_stats(f)
        print(f"  {f:<40}  server={n:>5}  v DB={db['count']:>4}")
    if len(small) > 20:
        print(f"  ... a ďalších {len(small)-20} priečinkov")


main()
