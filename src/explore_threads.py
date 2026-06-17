"""Exploratory thread summarization with llama3.1:8b.

Selects 5-6 coherent threads from different projects, cleans the text,
and asks llama3.1:8b an open-ended question about each thread.

Usage: python -m src.explore_threads
"""

import io
import os
import re
import sqlite3
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH    = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL  = "llama3.1:8b"

PROMPT = (
    "Toto je celé e-mailové vlákno z architektonickej firmy. "
    "Prečítaj ho a zhrň čo je v ňom podstatné — o čom to bolo, "
    "čo sa vyriešilo alebo rozhodlo, čo ostalo otvorené, "
    "a čo by stálo za zapamätanie do budúcna. "
    "Píš vecne a stručne.\n\n"
    "=== VLÁKNO ===\n{thread_text}\n=== KONIEC VLÁKNA ==="
)

MAX_BODY_CHARS  = 500   # per email
MAX_THREAD_CHARS = 9000  # total thread text sent to LLM
MAX_EMAILS_IN_THREAD = 18  # cap — long threads still readable


# ── text cleaning ──────────────────────────────────────────────────────────────

_QUOTE_LINE    = re.compile(r"^>+\s?")
_SIG_MARKERS   = re.compile(
    r"^(-{2,}|_{10,}|S pozdravom|Best regards|Regards,|Kind regards|Ing\.|"
    r"Sent from|Poslan[éo] z|This e-?mail|CONFIDENTIAL|DISCLAIMER)",
    re.IGNORECASE,
)
_FORWARD_HDR   = re.compile(
    r"^(Od|From|Komu|To|Predmet|Subject|Dátum|Date|Sent|CC|Cc):\s",
    re.IGNORECASE,
)
_FORWARD_SEP   = re.compile(r"^-{5,}\s*(Original|Forwarded|Pôvodná)", re.IGNORECASE)


def clean_body(text: str) -> str:
    if not text:
        return ""
    lines, sig = [], False
    for line in text.splitlines():
        s = line.strip()
        if not sig and (_SIG_MARKERS.match(s) or s in ("--", "— ")):
            sig = True
        if sig:
            continue
        if _QUOTE_LINE.match(s) or _FORWARD_HDR.match(line) or _FORWARD_SEP.match(s):
            continue
        lines.append(line)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return result


# ── thread selection ───────────────────────────────────────────────────────────

def _find_thread_for_keyword(conn, kw_condition: str,
                              min_emails: int = 5) -> dict | None:
    """Return the best thread_id (by email count) where ≥3 emails match kw."""
    rows = conn.execute(f"""
        SELECT thread_id, COUNT(*) AS cnt
        FROM emails
        WHERE {kw_condition} AND thread_id IS NOT NULL
          AND body_text IS NOT NULL AND body_text != ''
        GROUP BY thread_id
        HAVING cnt >= 3
        ORDER BY cnt DESC
        LIMIT 1
    """).fetchall()
    if not rows:
        return None
    tid = rows[0][0]
    return _load_thread(conn, tid, min_emails)


def _find_random_threads(conn, exclude_conds: list[str],
                         n: int = 2, min_emails: int = 6) -> list[dict]:
    excl = " AND NOT (" + " OR ".join(f"({c})" for c in exclude_conds) + ")"
    rows = conn.execute(f"""
        SELECT thread_id, COUNT(*) AS cnt
        FROM emails
        WHERE thread_id IS NOT NULL
          AND body_text IS NOT NULL AND body_text != ''
          {excl}
        GROUP BY thread_id
        HAVING cnt >= {min_emails}
        ORDER BY RANDOM()
        LIMIT {n * 5}
    """).fetchall()
    results = []
    seen = set()
    for tid, _ in rows:
        if tid in seen:
            continue
        t = _load_thread(conn, tid, min_emails)
        if t:
            results.append(t)
            seen.add(tid)
        if len(results) >= n:
            break
    return results


def _load_thread(conn, thread_id: str, min_emails: int = 3) -> dict | None:
    rows = conn.execute("""
        SELECT id, subject, from_address, to_addresses, date, body_text
        FROM emails
        WHERE thread_id = ? AND body_text IS NOT NULL AND body_text != ''
        ORDER BY date
    """, (thread_id,)).fetchall()
    if len(rows) < min_emails:
        return None
    emails = [{"id": r[0], "subject": r[1] or "", "from": r[2] or "",
               "to": r[3] or "", "date": r[4] or "", "body": r[5] or ""}
              for r in rows]
    participants = list(dict.fromkeys(e["from"] for e in emails if "@" in e["from"]))
    return {
        "thread_id": thread_id,
        "subject":   emails[0]["subject"],
        "emails":    emails,
        "n":         len(emails),
        "date_from": emails[0]["date"][:10],
        "date_to":   emails[-1]["date"][:10],
        "participants": participants[:8],
    }


# ── thread text builder ────────────────────────────────────────────────────────

def build_thread_text(thread: dict) -> str:
    emails = thread["emails"]
    # cap at MAX_EMAILS_IN_THREAD, prefer first + last emails
    if len(emails) > MAX_EMAILS_IN_THREAD:
        half = MAX_EMAILS_IN_THREAD // 2
        emails = emails[:half] + emails[-half:]
    parts = []
    for em in emails:
        date_str = em["date"][:16] if em["date"] else "?"
        header   = f"[{date_str}] {em['from']} → {(em['to'] or '')[:60]}"
        body     = clean_body(em["body"])
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + " [...]"
        parts.append(f"{header}\n{body}")
    text = "\n\n---\n\n".join(parts)
    if len(text) > MAX_THREAD_CHARS:
        text = text[:MAX_THREAD_CHARS] + "\n\n[... vlákno ďalej skrátené ...]"
    return text


# ── LLM call ──────────────────────────────────────────────────────────────────

def llm_summarize(thread_text: str) -> str:
    prompt = PROMPT.format(thread_text=thread_text)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[CHYBA LLM: {e}]"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)

    # ── define target categories ──────────────────────────────────────────────
    categories = [
        ("Westend",        "LOWER(subject) LIKE '%westend%'"),
        ("Patronka/2202",  "(LOWER(subject) LIKE '%patronka%' OR subject LIKE '%2202%')"),
        ("Eurovea/Tower",  "(LOWER(subject) LIKE '%eurovea%' OR LOWER(subject) LIKE '%tower%')"),
        ("Svetlotechnika", "LOWER(subject) LIKE '%svetlotechn%'"),
    ]

    threads = []
    exclude_conds = [c for _, c in categories]

    for label, cond in categories:
        t = _find_thread_for_keyword(conn, cond)
        if t:
            t["label"] = label
            threads.append(t)
            print(f"  [{label}] '{t['subject'][:55]}'  n={t['n']}  "
                  f"{t['date_from']}–{t['date_to']}", flush=True)
        else:
            print(f"  [{label}] nenajdene vlakno s >=5 emailov", flush=True)

    # 2 random long threads
    randoms = _find_random_threads(conn, exclude_conds, n=2, min_emails=7)
    for t in randoms:
        t["label"] = "Nahodne"
        threads.append(t)
        print(f"  [Nahodne] '{t['subject'][:55]}'  n={t['n']}  "
              f"{t['date_from']}–{t['date_to']}", flush=True)

    conn.close()

    # ── process each thread ───────────────────────────────────────────────────
    for i, thread in enumerate(threads, 1):
        print(f"\n{'='*72}")
        print(f"  VLAKNO {i}/{len(threads)}: [{thread['label']}]")
        print(f"  Predmet   : {thread['subject']}")
        print(f"  Pocet     : {thread['n']} emailov")
        print(f"  Rozsah    : {thread['date_from']} – {thread['date_to']}")
        print(f"  Ucastnici : {', '.join(thread['participants'])}")
        print(f"{'='*72}")

        thread_text = build_thread_text(thread)

        print(f"\n  --- VLAKNO (skratene, {len(thread_text)} znakov) ---")
        # Print first 2 emails as preview
        preview_emails = thread["emails"][:2]
        for em in preview_emails:
            date_str = em["date"][:16]
            body_preview = clean_body(em["body"])[:200].replace("\n", " ")
            print(f"  [{date_str}] {em['from']}")
            print(f"    {body_preview}...")
        if thread["n"] > 2:
            print(f"  ... (+{thread['n']-2} dalsich mailov) ...")

        print(f"\n  --- LLM ({LLM_MODEL}) ---", flush=True)
        summary = llm_summarize(thread_text)
        print(f"{summary}\n")


if __name__ == "__main__":
    main()
