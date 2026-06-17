"""Structured JSON extraction from email threads via llama3.1:8b.

Tests whether the LLM reliably fills a fixed schema:
  o_com_to_bolo / ucastnici / ulohy / rozhodnutia / otvorene / parametre

Usage: python -m src.extract_threads
"""

import io
import json
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

PROMPT = """\
Si expert asistent na spracovanie emailov z architektonickej firmy.
Prečítaj nasledujúce emailové vlákno a extrahuj štruktúrované informácie.

Vráť VÝLUČNE validný JSON objekt v tomto presnom formáte (nič iné okrem JSON):

{{
  "o_com_to_bolo": [
    "odsek 1 (téma A)",
    "odsek 2 (téma B, ak vlákno obsahuje viacero nezávislých tém)"
  ],
  "ucastnici": [
    {{"meno": "...", "strana": "GFI / JTRE / externý-<doména> / iné"}}
  ],
  "ulohy": [
    {{"co": "...", "kto_ma_spravit": "...", "zadal": "...", "termin": "... alebo null"}}
  ],
  "rozhodnutia": ["uzavreté rozhodnutie 1"],
  "otvorene": ["čo ostalo visieť 1"],
  "parametre": ["termín / cena / číslo s kontextom"]
}}

Pravidlá:
- "o_com_to_bolo": viac odsekov ak vlákno rieši viacero nezávislých tém, jeden odsek ak je súdržné
- ak pole nemá obsah, vráť prázdny zoznam []
- nevymýšľaj informácie ktoré v texte nie sú
- píš vecne po slovensky
- vráť IBA JSON bez akéhokoľvek vysvetlenia alebo markdown

=== VLÁKNO ===
{thread_text}
=== KONIEC VLÁKNA ==="""

MAX_BODY_CHARS   = 500
MAX_THREAD_CHARS = 9000
MAX_EMAILS       = 18

CATEGORIES = [
    ("Westend",        "LOWER(subject) LIKE '%westend%'"),
    ("Patronka/2202",  "(LOWER(subject) LIKE '%patronka%' OR subject LIKE '%2202%')"),
    ("ISTER TOWER",    "(LOWER(subject) LIKE '%ister tower%' OR LOWER(subject) LIKE '%kalos%')"),
    ("Svetlotechnika", "LOWER(subject) LIKE '%svetlotechn%'"),
    ("MSH",            "LOWER(subject) LIKE '%msh%'"),
    ("Pulsar",         "LOWER(subject) LIKE '%pulsar%'"),
]


# ── cleaning ───────────────────────────────────────────────────────────────────

_SIG = re.compile(
    r"^(-{2,}|_{10,}|S pozdravom|Best regards|Regards,|Kind regards|"
    r"Sent from|Poslan[éo] z|This e-?mail|CONFIDENTIAL|DISCLAIMER)",
    re.IGNORECASE,
)
_FWDHDR = re.compile(r"^(Od|From|Komu|To|Predmet|Subject|Dátum|Date|Sent|CC|Cc):\s",
                     re.IGNORECASE)
_FWDSEP = re.compile(r"^-{5,}\s*(Original|Forwarded|Pôvodná)", re.IGNORECASE)


def clean_body(text: str) -> str:
    lines, sig = [], False
    for line in text.splitlines():
        s = line.strip()
        if not sig and (_SIG.match(s) or s in ("--", "— ")):
            sig = True
        if sig or s.startswith(">") or _FWDHDR.match(line) or _FWDSEP.match(s):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ── thread loading ─────────────────────────────────────────────────────────────

def find_thread(conn, kw_cond: str, min_match: int = 3) -> dict | None:
    row = conn.execute(f"""
        SELECT thread_id, COUNT(*) AS cnt
        FROM emails
        WHERE {kw_cond} AND thread_id IS NOT NULL
          AND body_text IS NOT NULL AND body_text != ''
        GROUP BY thread_id HAVING cnt >= {min_match}
        ORDER BY cnt DESC LIMIT 1
    """).fetchone()
    if not row:
        return None
    tid = row[0]
    emails = conn.execute("""
        SELECT id, subject, from_address, to_addresses, date, body_text
        FROM emails WHERE thread_id = ?
          AND body_text IS NOT NULL AND body_text != ''
        ORDER BY date
    """, (tid,)).fetchall()
    if len(emails) < 3:
        return None
    em_list = [{"id": r[0], "subject": r[1] or "", "from": r[2] or "",
                "to": r[3] or "", "date": r[4] or "", "body": r[5] or ""}
               for r in emails]
    participants = list(dict.fromkeys(e["from"] for e in em_list if "@" in e["from"]))
    return {
        "subject": em_list[0]["subject"],
        "emails":  em_list,
        "n":       len(em_list),
        "date_from": em_list[0]["date"][:10],
        "date_to":   em_list[-1]["date"][:10],
        "participants": participants[:8],
    }


def build_thread_text(thread: dict) -> str:
    emails = thread["emails"]
    if len(emails) > MAX_EMAILS:
        h = MAX_EMAILS // 2
        emails = emails[:h] + emails[-h:]
    parts = []
    for em in emails:
        header = f"[{em['date'][:16]}] {em['from']} → {(em['to'] or '')[:60]}"
        body   = clean_body(em["body"])
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + " [...]"
        parts.append(f"{header}\n{body}")
    text = "\n\n---\n\n".join(parts)
    if len(text) > MAX_THREAD_CHARS:
        text = text[:MAX_THREAD_CHARS] + "\n\n[... skrátené ...]"
    return text


# ── LLM ───────────────────────────────────────────────────────────────────────

def llm_extract(thread_text: str) -> tuple[dict | None, str]:
    prompt = PROMPT.format(thread_text=thread_text)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
    except Exception as e:
        return None, f"[CHYBA HTTP: {e}]"

    # strip optional markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # find the outermost {...}
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)

    try:
        return json.loads(cleaned), raw
    except json.JSONDecodeError as e:
        return None, f"[JSON PARSE ERROR: {e}]\n{raw}"


# ── pretty print ───────────────────────────────────────────────────────────────

def print_extract(label: str, thread: dict, data: dict | None, raw: str):
    print(f"\n{'='*72}")
    print(f"  [{label}]  \"{thread['subject'][:60]}\"")
    print(f"  {thread['n']} emailov  {thread['date_from']} – {thread['date_to']}")
    print(f"  Účastníci: {', '.join(thread['participants'])}")
    print(f"{'='*72}")

    if data is None:
        print(f"\n  !! LLM nevratil validny JSON:\n{raw}\n")
        return

    def section(title, items):
        if not items:
            print(f"\n  {title}: (prázdne)")
            return
        print(f"\n  {title}:")
        for item in items:
            if isinstance(item, dict):
                parts = "  |  ".join(f"{k}: {v}" for k, v in item.items() if v)
                print(f"    • {parts}")
            else:
                print(f"    • {item}")

    section("O čom to bolo",  data.get("o_com_to_bolo", []))
    section("Účastníci",       data.get("ucastnici", []))
    section("Úlohy",           data.get("ulohy", []))
    section("Rozhodnutia",     data.get("rozhodnutia", []))
    section("Otvorené",        data.get("otvorene", []))
    section("Parametre",       data.get("parametre", []))
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)

    print("=== Structured extraction test — 6 vlákien ===\n")

    for label, cond in CATEGORIES:
        thread = find_thread(conn, cond)
        if not thread:
            print(f"  [{label}] NENAJDENE\n")
            continue
        print(f"  [{label}] '{thread['subject'][:55]}'  n={thread['n']}  "
              f"{thread['date_from']}–{thread['date_to']}", flush=True)

        thread_text = build_thread_text(thread)
        data, raw   = llm_extract(thread_text)
        print_extract(label, thread, data, raw)

    conn.close()
    print("=== DONE ===")


if __name__ == "__main__":
    main()
