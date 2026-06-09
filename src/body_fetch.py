"""IMAP body fetch — downloads email bodies into body_text, body_snippet, language."""

import argparse
import imaplib
import logging
import os
import re
import sys
from collections import defaultdict
from email import message_from_bytes
from pathlib import Path

from dotenv import load_dotenv

try:
    from .db import get_connection, get_emails_without_body
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.db import get_connection, get_emails_without_body

load_dotenv()

IMAP_HOST = os.environ["IMAP_HOST"]
IMAP_PORT = int(os.environ.get("IMAP_PORT", 993))
IMAP_USER = os.environ["IMAP_USER"]
IMAP_PASS = os.environ["IMAP_PASS"]
DB_PATH   = os.environ.get("DB_PATH", "data/emails.db")

BATCH_SIZE = 50

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _err_log() -> logging.Logger:
    err = logging.getLogger("body_fetch.errors")
    if not err.handlers:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            Path(DB_PATH).parent / "errors_body.log", encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        err.addHandler(fh)
        err.propagate = False
    return err


# ── text extraction ───────────────────────────────────────────────────────────

def _decode_part(payload: bytes, charset: str | None) -> str:
    for enc in filter(None, [charset, "utf-8", "windows-1250", "latin-1"]):
        try:
            return payload.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return payload.decode("latin-1", errors="replace")


def _strip_html(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html,
                  flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    for entity, char in [("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"),
                         ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'")]:
        html = html.replace(entity, char)
    return re.sub(r"\s+", " ", html).strip()


def _extract_text(msg) -> str:
    """Return plain text from a Message object: prefer text/plain, fallback text/html."""
    plain: list[str] = []
    html:  list[str] = []

    for part in msg.walk():
        if part.get_content_disposition() == "attachment":
            continue
        ct = part.get_content_type()
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        charset = part.get_content_charset()

        if ct == "text/plain":
            plain.append(_decode_part(payload, charset))
        elif ct == "text/html":
            html.append(_decode_part(payload, charset))

    if plain:
        return re.sub(r"\s+", " ", " ".join(plain)).strip()
    if html:
        return _strip_html(" ".join(html))
    return ""


# ── language detection ────────────────────────────────────────────────────────

# Characters unique to Slovak (not in German or English)
_SK_UNIQUE = frozenset("ľĺŕôĽĹŔÔ")
# Slovak/Czech diacritics (shared but strong signal vs. English)
_SK_COMMON = frozenset("áčďéíňóšťúýžÁČĎÉÍŇÓŠŤÚÝŽ")
# German-specific (not common in Slovak)
_DE_UNIQUE = frozenset("üößÜÖ")


def detect_language(text: str) -> str:
    if not text or len(text) < 15:
        return "other"

    sk_unique = sum(1 for c in text if c in _SK_UNIQUE)
    if sk_unique >= 1:
        return "sk"

    sk_common = sum(1 for c in text if c in _SK_COMMON)
    de_unique  = sum(1 for c in text if c in _DE_UNIQUE)

    total = len(text)
    sk_ratio = sk_common / total
    de_ratio = de_unique  / total

    if sk_ratio > 0.015:
        return "sk"
    if de_ratio > 0.01:
        return "de"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / total
    if ascii_ratio > 0.96:
        return "en"
    return "other"


# ── IMAP helpers ──────────────────────────────────────────────────────────────

def _connect() -> imaplib.IMAP4_SSL:
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    imap.login(IMAP_USER, IMAP_PASS)
    log.info("Connected to %s", IMAP_HOST)
    return imap


def _fetch_bodies_batch(imap: imaplib.IMAP4_SSL,
                        uids: list[int]) -> dict[int, bytes]:
    """FETCH BODY.PEEK[] for a list of UIDs. Returns {uid: raw_bytes}."""
    uid_set = b",".join(str(u).encode() for u in uids)
    status, data = imap.uid("FETCH", uid_set, "(BODY.PEEK[])")
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

def _update_bodies(conn, rows: list[tuple]) -> None:
    """Batch UPDATE: rows = list of (body_text, body_snippet, language, email_id)."""
    conn.executemany(
        "UPDATE emails SET body_text=?, body_snippet=?, language=? WHERE id=?",
        rows,
    )
    conn.commit()


# ── core sync logic ───────────────────────────────────────────────────────────

def fetch_folder_bodies(
    imap: imaplib.IMAP4_SSL,
    conn,
    folder: str,
    pending: list,          # list of sqlite3.Row (id, imap_uid, folder, subject, message_id)
    progress_offset: int,
    total_overall: int,
) -> dict:
    err = _err_log()
    stats = {"ok": 0, "empty": 0, "error": 0}

    status, _ = imap.select(f'"{folder}"', readonly=True)
    if status != "OK":
        log.warning("Cannot SELECT %r — skipping", folder)
        err.error("Cannot SELECT folder %r", folder)
        stats["error"] += len(pending)
        return stats

    processed = 0
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i : i + BATCH_SIZE]
        uid_to_row = {row["imap_uid"]: row for row in batch}
        uids = list(uid_to_row.keys())

        try:
            bodies = _fetch_bodies_batch(imap, uids)
        except Exception as exc:
            err.error("FETCH batch folder=%s uids=%s: %s", folder, uids[:3], exc)
            stats["error"] += len(batch)
            processed += len(batch)
            continue

        updates: list[tuple] = []
        for uid, row in uid_to_row.items():
            raw = bodies.get(uid)
            if raw is None:
                err.error("UID %d folder=%s: no data", uid, folder)
                stats["error"] += 1
                continue
            try:
                msg  = message_from_bytes(raw)
                text = _extract_text(msg)
                body_text    = text[:1000] if text else ""
                body_snippet = text[:150]  if text else ""
                lang = detect_language(body_text)
                updates.append((body_text, body_snippet, lang, row["id"]))
                if text:
                    stats["ok"] += 1
                else:
                    stats["empty"] += 1
            except Exception as exc:
                err.error("Parse UID %d folder=%s: %s", uid, folder, exc)
                stats["error"] += 1

        if updates:
            _update_bodies(conn, updates)

        processed += len(batch)
        log.info("[%d/%d] %s  ok=%d empty=%d err=%d",
                 progress_offset + processed, total_overall, folder,
                 stats["ok"], stats["empty"], stats["error"])

    return stats


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch email bodies from IMAP into DB")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Max emails to process (default: all)")
    parser.add_argument("--folder", default=None,
                        help="Only process this folder")
    args = parser.parse_args()

    conn = get_connection(DB_PATH)
    pending_all = get_emails_without_body(DB_PATH, limit=args.limit or 99999)

    if args.folder:
        pending_all = [r for r in pending_all if r["folder"] == args.folder]

    if not pending_all:
        log.info("Ziadne emaily bez body_text — vsetko je uz spracovane.")
        conn.close()
        return

    # group by folder
    by_folder: dict[str, list] = defaultdict(list)
    for row in pending_all:
        by_folder[row["folder"]].append(row)

    total = len(pending_all)
    log.info("Spracujem %d emailov v %d priecinku/och", total, len(by_folder))

    imap = _connect()
    all_stats: dict = {"ok": 0, "empty": 0, "error": 0}
    offset = 0

    for folder, rows in by_folder.items():
        s = fetch_folder_bodies(imap, conn, folder, rows, offset, total)
        offset += len(rows)
        for k in all_stats:
            all_stats[k] += s[k]

    imap.logout()
    conn.close()

    print("\n--- Body fetch summary ----------------------------------------")
    print(f"  ok (text najdeny)   : {all_stats['ok']:>6}")
    print(f"  empty (bez textu)   : {all_stats['empty']:>6}")
    print(f"  errors              : {all_stats['error']:>6}")
    print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
