"""IMAP sync module — downloads email metadata from server to local DB."""

import argparse
import imaplib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from email import message_from_bytes
from pathlib import Path

_IMAP_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

from dotenv import load_dotenv

from .db import get_connection, init_db
from .utils import (
    decode_header_value,
    extract_attachments,
    extract_thread_id,
    parse_address_list,
    parse_date,
)

load_dotenv()

IMAP_HOST = os.environ["IMAP_HOST"]
IMAP_PORT = int(os.environ.get("IMAP_PORT", 993))
IMAP_USER = os.environ["IMAP_USER"]
IMAP_PASS = os.environ["IMAP_PASS"]
DB_PATH   = os.environ.get("DB_PATH", "data/emails.db")

BATCH_SIZE      = 100
DEFAULT_FOLDERS = ["INBOX", "Sent Items"]

# ── logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _error_logger(db_path: str) -> logging.Logger:
    err_log = logging.getLogger("sync.errors")
    if not err_log.handlers:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            Path(db_path).parent / "errors.log", encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        err_log.addHandler(fh)
        err_log.propagate = False
    return err_log


# ── IMAP helpers ─────────────────────────────────────────────────────────────

def _connect() -> imaplib.IMAP4_SSL:
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    imap.login(IMAP_USER, IMAP_PASS)
    log.info("Connected to %s as %s", IMAP_HOST, IMAP_USER)
    return imap


def _parse_folder_name(line: bytes) -> str | None:
    """Extract folder name from a LIST response line."""
    # format: (\Flags) "delim" "Folder Name"  or  (\Flags) "delim" FolderName
    try:
        decoded = line.decode("utf-8", errors="replace")
        # quoted name
        m = re.search(r'"([^"]+)"\s*$', decoded)
        if m:
            return m.group(1)
        # unquoted name (no spaces)
        m = re.search(r'\)\s+\S+\s+(\S+)\s*$', decoded)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def list_folders(imap: imaplib.IMAP4_SSL) -> list[str]:
    status, lines = imap.list()
    if status != "OK":
        raise RuntimeError("IMAP LIST failed")
    folders = []
    for line in lines:
        if isinstance(line, bytes):
            name = _parse_folder_name(line)
            if name:
                folders.append(name)
    return folders


def _to_imap_date(iso_date: str) -> str:
    """Convert 'YYYY-MM-DD' to IMAP SEARCH date format 'DD-Mon-YYYY'."""
    d = datetime.strptime(iso_date, "%Y-%m-%d")
    return f"{d.day:02d}-{_IMAP_MONTHS[d.month - 1]}-{d.year}"


def _search_new_uids(imap: imaplib.IMAP4_SSL, last_uid: int,
                     since: str | None = None) -> list[int]:
    """Return UIDs to fetch, sorted ascending.

    - last_uid > 0  : incremental sync, ignores since (already filtered by UID)
    - last_uid == 0 : first sync; uses SINCE date if provided, else ALL
    """
    if last_uid > 0:
        criterion = f"UID {last_uid + 1}:*"
    elif since:
        criterion = f"SINCE {_to_imap_date(since)}"
    else:
        criterion = "ALL"

    status, data = imap.uid("SEARCH", None, criterion)
    if status != "OK" or not data or data[0] == b"":
        return []
    uids = [int(u) for u in data[0].split()]
    return [u for u in uids if u > last_uid]


def _fetch_headers_batch(
    imap: imaplib.IMAP4_SSL, uids: list[int]
) -> dict[int, bytes]:
    """Fetch HEADER bytes for a list of UIDs. Returns {uid: raw_header_bytes}."""
    uid_set = b",".join(str(u).encode() for u in uids)
    status, data = imap.uid("FETCH", uid_set, "(BODY.PEEK[HEADER] RFC822.SIZE)")
    if status != "OK":
        return {}

    results: dict[int, bytes] = {}
    for item in data:
        if isinstance(item, tuple):
            meta, raw = item
            m = re.search(rb"UID (\d+)", meta)
            if m:
                results[int(m.group(1))] = raw
    return results


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_last_uid(conn, folder: str) -> int:
    row = conn.execute(
        "SELECT last_uid FROM sync_state WHERE folder = ?", (folder,)
    ).fetchone()
    return row["last_uid"] if row else 0


def _get_known_uids(conn, folder: str) -> set[int]:
    """Return all imap_uid values already stored in DB for this folder."""
    rows = conn.execute(
        "SELECT imap_uid FROM emails WHERE folder = ? AND imap_uid IS NOT NULL",
        (folder,),
    ).fetchall()
    return {r["imap_uid"] for r in rows}


def _save_sync_state(conn, folder: str, last_uid: int) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute("""
        INSERT INTO sync_state (folder, last_uid, last_sync)
        VALUES (?, ?, ?)
        ON CONFLICT(folder) DO UPDATE SET
            last_uid  = excluded.last_uid,
            last_sync = excluded.last_sync
    """, (folder, last_uid, now))
    conn.commit()


def _insert_email(conn, row: dict) -> bool:
    """Insert one email row. Returns True on insert, False if already exists."""
    try:
        conn.execute("""
            INSERT OR IGNORE INTO emails (
                message_id, thread_id, in_reply_to, "references",
                date, from_address, from_name,
                to_addresses, cc_addresses, subject,
                folder, has_attachments, attachment_names, attachment_types,
                size_bytes, imap_uid, synced_at
            ) VALUES (
                :message_id, :thread_id, :in_reply_to, :references,
                :date, :from_address, :from_name,
                :to_addresses, :cc_addresses, :subject,
                :folder, :has_attachments, :attachment_names, :attachment_types,
                :size_bytes, :imap_uid, :synced_at
            )
        """, row)
        return conn.execute("SELECT changes()").fetchone()[0] == 1
    except Exception:
        raise


# ── header → dict ─────────────────────────────────────────────────────────────

def _parse_email_row(uid: int, folder: str, raw: bytes) -> dict:
    msg = message_from_bytes(raw)

    froms     = parse_address_list(msg.get("From", ""))
    from_addr = froms[0]["address"] if froms else ""
    from_name = froms[0]["name"]    if froms else ""

    to_list  = parse_address_list(msg.get("To",  ""))
    cc_list  = parse_address_list(msg.get("Cc",  ""))

    message_id  = decode_header_value(msg.get("Message-ID",  "")).strip()
    in_reply_to = decode_header_value(msg.get("In-Reply-To", "")).strip() or None
    refs        = decode_header_value(msg.get("References",  "")).strip() or None

    att_names, att_types = extract_attachments(msg)

    # Heuristic: walk() finds nothing with header-only fetch, so check top-level
    # Content-Type. multipart/mixed reliably indicates file attachments.
    # multipart/related = HTML with embedded images (not user-visible files).
    # multipart/alternative = text+HTML variants, no attachments.
    content_type = (msg.get("Content-Type") or "").lower()
    if att_names:
        has_att = 1
    elif "multipart/mixed" in content_type:
        has_att = 1
    else:
        has_att = 0

    return {
        "message_id":       message_id or f"<no-id-uid-{uid}@{folder}>",
        "thread_id":        extract_thread_id(message_id, in_reply_to, refs) or None,
        "in_reply_to":      in_reply_to,
        "references":       refs,
        "date":             parse_date(msg.get("Date")),
        "from_address":     from_addr,
        "from_name":        from_name,
        "to_addresses":     json.dumps(to_list,  ensure_ascii=False),
        "cc_addresses":     json.dumps(cc_list,  ensure_ascii=False),
        "subject":          decode_header_value(msg.get("Subject", "")),
        "folder":           folder,
        "has_attachments":  has_att,
        "attachment_names": json.dumps(att_names, ensure_ascii=False),
        "attachment_types": json.dumps(att_types, ensure_ascii=False),
        "size_bytes":       len(raw),
        "imap_uid":         uid,
        "synced_at":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── sync one folder ───────────────────────────────────────────────────────────

def sync_folder(
    imap: imaplib.IMAP4_SSL,
    conn,
    folder: str,
    limit: int | None = None,
    since: str | None = None,
    backfill: bool = False,
) -> dict:
    err_log = _error_logger(DB_PATH)
    stats = {"inserted": 0, "skipped": 0, "errors": 0, "folder": folder}

    status, _ = imap.select(f'"{folder}"', readonly=True)
    if status != "OK":
        log.error("Cannot SELECT folder %r", folder)
        return stats

    last_uid = _get_last_uid(conn, folder)

    if backfill:
        # Fetch ALL UIDs from server, subtract what we already have in DB.
        # Resume-safe: on restart, known_uids grows and we skip already-fetched.
        status2, data = imap.uid("SEARCH", None, "ALL")
        if status2 != "OK" or not data or data[0] == b"":
            server_uids: list[int] = []
        else:
            server_uids = sorted(int(u) for u in data[0].split())

        known_uids = _get_known_uids(conn, folder)
        all_uids   = [u for u in server_uids if u not in known_uids]
        log.info(
            "%-30s  backfill: %d on server, %d in DB, %d to fetch",
            folder, len(server_uids), len(known_uids), len(all_uids),
        )
    else:
        all_uids = _search_new_uids(imap, last_uid, since=since)

    if limit:
        all_uids = all_uids[:limit]

    total = len(all_uids)
    if total == 0:
        log.info("%-30s  nothing to fetch (last_uid=%d)", folder, last_uid)
        return stats

    log.info("%-30s  %d UIDs to fetch", folder, total)

    processed      = 0
    batch_last_uid = last_uid

    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_uids[batch_start : batch_start + BATCH_SIZE]

        try:
            headers_map = _fetch_headers_batch(imap, batch)
        except Exception as exc:
            log.error("FETCH failed for batch at index %d: %s", batch_start, exc)
            err_log.error("FETCH batch %d-%d folder=%s: %s",
                          batch[0], batch[-1], folder, exc)
            stats["errors"] += len(batch)
            continue

        for uid in batch:
            raw = headers_map.get(uid)
            if raw is None:
                err_log.error("UID %d folder=%s: no data returned", uid, folder)
                stats["errors"] += 1
                continue
            try:
                row = _parse_email_row(uid, folder, raw)
                inserted = _insert_email(conn, row)
                if inserted:
                    stats["inserted"] += 1
                else:
                    stats["skipped"] += 1
                if uid > batch_last_uid:
                    batch_last_uid = uid
            except Exception as exc:
                err_log.error("UID %d folder=%s: %s", uid, folder, exc)
                stats["errors"] += 1

        conn.commit()
        # In backfill mode: only advance last_uid if we saw a higher UID than before
        # (never lower the watermark — incremental sync must still work after backfill)
        if not backfill or batch_last_uid > last_uid:
            _save_sync_state(conn, folder, batch_last_uid)

        processed += len(batch)
        log.info("[%d/%d] %s  (+%d inserted, %d skipped, %d errors)",
                 processed, total, folder,
                 stats["inserted"], stats["skipped"], stats["errors"])

    return stats


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync email metadata from IMAP to SQLite")
    parser.add_argument("--folder", help="Sync only this folder")
    parser.add_argument("--limit",  type=int, help="Max UIDs to fetch per folder (for testing)")
    parser.add_argument("--since",  default="2024-06-08",
                        help="Fetch emails since this date for first sync, format YYYY-MM-DD (default: 2024-06-08)")
    parser.add_argument("--list",     action="store_true", help="List available folders and exit")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch ALL UIDs in folder, skipping what is already in DB "
                             "(ignores --since and last_uid; safe to interrupt and resume)")
    args = parser.parse_args()

    init_db(DB_PATH)
    conn = get_connection(DB_PATH)
    imap = _connect()

    available = list_folders(imap)

    if args.list:
        print("\nAvailable IMAP folders:")
        for f in available:
            print(f"  {f}")
        imap.logout()
        conn.close()
        return

    log.info("Found %d folders on server", len(available))

    if args.folder:
        folders_to_sync = [args.folder]
    else:
        # sync DEFAULT_FOLDERS that actually exist on the server
        available_lower = {f.lower(): f for f in available}
        folders_to_sync = []
        for target in DEFAULT_FOLDERS:
            match = available_lower.get(target.lower())
            if match:
                folders_to_sync.append(match)
            else:
                log.warning("Folder %r not found on server — skipping", target)

    if args.backfill:
        log.info("Mode: BACKFILL (fetches all missing UIDs, skips existing by message_id)")
    else:
        log.info("Date filter for first sync: SINCE %s", args.since)

    all_stats: list[dict] = []
    for folder in folders_to_sync:
        stats = sync_folder(
            imap, conn, folder,
            limit    = args.limit,
            since    = args.since,
            backfill = args.backfill,
        )
        all_stats.append(stats)

    imap.logout()
    conn.close()

    print("\n--- Sync summary -------------------------------------------")
    total_inserted = total_skipped = total_errors = 0
    for s in all_stats:
        print(f"  {s['folder']:<30}  inserted={s['inserted']:>6}  "
              f"skipped={s['skipped']:>6}  errors={s['errors']:>4}")
        total_inserted += s["inserted"]
        total_skipped  += s["skipped"]
        total_errors   += s["errors"]
    print(f"  {'TOTAL':<30}  inserted={total_inserted:>6}  "
          f"skipped={total_skipped:>6}  errors={total_errors:>4}")
    print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
