"""Two-step conversation extraction: free summary → identification layer.

Step 1: llama3.1:8b reads cleaned thread text → free Slovak summary
Step 2: llama3.1:8b reads summary → topic/project/participant identification

Usage: python -m src.conv_extract
"""

import io
import os
import re
import sqlite3
import sys
import unicodedata

import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH    = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL  = "llama3.1:8b"

# 8 conversations: (conversation_id, label)
SELECTED = [
    # ── large project ──────────────────────────────────────────────────────────
    (11522, "Westend režim",            "large"),
    (4836,  "ISTER TOWER / KALOS",      "large"),
    (13753, "One Eurovea design mtg",   "large"),
    # ── medium ────────────────────────────────────────────────────────────────
    (967,   "Skypark terasy",           "medium"),
    (2718,  "Špitálska – model foto",   "medium"),
    (8136,  "Gehl + GFI Bratislava",    "medium"),
    # ── divergent ─────────────────────────────────────────────────────────────
    (10169, "AI diskusia (2-ročná)",    "divergent"),
    (852,   "Klingerka (4 roky)",       "divergent"),
]

MAX_EMAILS_SAMPLED  = 18
MAX_BODY_CHARS      = 450
MAX_THREAD_CHARS    = 9000

PROMPT_STEP1 = """\
Toto je e-mailová konverzácia z architektonickej firmy.
Prečítaj ju a napíš vecné zhrnutie o čom bola — čo sa riešilo, čo sa rozhodlo, \
čo ostalo otvorené.
Ak konverzácia rieši viacero nezávislých vecí, rozdeľ zhrnutie na odseky podľa tém.
Píš po slovensky, vecne.

=== KONVERZÁCIA ===
{thread_text}
=== KONIEC ==="""

PROMPT_STEP2 = """\
Z tohto zhrnutia e-mailovej konverzácie vytiahni:
(1) Témy — čoho sa konverzácia týkala (1-5 krátkych tém)
(2) Projekt(y) — identifikovateľné projekty alebo akcie ak existujú
(3) Účastníci — mená / e-maily a ich strana (GFI / JTRE / externý)

Vráť ako jednoduchý zoznam s odrážkami, bez dlhých popisov.

=== ZHRNUTIE ===
{summary}
=== KONIEC ==="""

# ── cleaning ───────────────────────────────────────────────────────────────────

_SIG = re.compile(
    r"^(-{2,}|_{10,}|S pozdravom|Best regards|Regards,|Kind regards|"
    r"Sent from|Poslan[éo] z|This e-?mail|CONFIDENTIAL|DISCLAIMER)",
    re.IGNORECASE,
)
_FWDHDR = re.compile(r"^(Od|From|Komu|To|Predmet|Subject|Dátum|Date|Sent|CC|Cc):\s",
                     re.IGNORECASE)
_FWDSEP = re.compile(r"^-{5,}\s*(Original|Forwarded|Pôvodná)", re.IGNORECASE)
_PREFIX = re.compile(
    r"^(Re|RE|Fwd|FW|Fw|Odp|Odp\.|Odpoveď|VS|AW|Pfwd)\s*[:\s]\s*",
    re.IGNORECASE,
)


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


def norm_subj(s: str) -> str:
    s = (s or "").strip()
    while True:
        m = _PREFIX.match(s)
        if m:
            s = s[m.end():].strip()
        else:
            break
    return s


# ── thread text builder ────────────────────────────────────────────────────────

def build_thread_text(emails: list[dict]) -> str:
    """Sample up to MAX_EMAILS_SAMPLED and build cleaned text."""
    n = len(emails)
    if n <= MAX_EMAILS_SAMPLED:
        sampled = emails
    else:
        # first 6, middle 6, last 6
        chunk = MAX_EMAILS_SAMPLED // 3
        mid_start = n // 2 - chunk // 2
        sampled = (emails[:chunk]
                   + emails[mid_start: mid_start + chunk]
                   + emails[-chunk:])
        sampled = sorted({e["id"]: e for e in sampled}.values(),
                         key=lambda x: x["date"] or "")

    parts = []
    for em in sampled:
        dt = (em["date"] or "")[:16]
        frm = em["from_address"] or "?"
        to  = (em["to_addresses"] or "")[:50]
        body = clean_body(em["body_text"] or "")
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + " [...]"
        if not body:
            continue
        parts.append(f"[{dt}] {frm} → {to}\n{body}")

    text = "\n\n---\n\n".join(parts)
    if len(text) > MAX_THREAD_CHARS:
        text = text[:MAX_THREAD_CHARS] + "\n\n[... skrátené ...]"
    return text


# ── LLM ───────────────────────────────────────────────────────────────────────

def llm(prompt: str, label: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[CHYBA {label}: {e}]"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    for conv_id, label, kind in SELECTED:
        # load emails
        rows = conn.execute("""
            SELECT id, subject, from_address, to_addresses, date, body_text
            FROM emails
            WHERE conversation_id = ?
              AND body_text IS NOT NULL AND body_text != ''
            ORDER BY date
        """, (conv_id,)).fetchall()
        emails = [dict(r) for r in rows]

        if not emails:
            print(f"\n[{label}] ŽIADNE EMAILY S TELOM — preskočené\n")
            continue

        # meta
        all_rows = conn.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM emails WHERE conversation_id=?",
            (conv_id,)
        ).fetchone()
        total_n    = all_rows[0]
        date_from  = (all_rows[1] or "")[:10]
        date_to    = (all_rows[2] or "")[:10]
        participants = list(dict.fromkeys(
            e["from_address"] for e in emails if e.get("from_address")
        ))[:6]
        first_subj = norm_subj(emails[0]["subject"] or "")

        print(f"\n{'='*72}")
        print(f"  [{kind.upper()}]  conv_id={conv_id}  \"{label}\"")
        print(f"  Subject    : {first_subj[:60]}")
        print(f"  Emailov    : {total_n}  (z toho s telom: {len(emails)})"
              f"  [{date_from} – {date_to}]")
        print(f"  Účastníci  : {', '.join(participants)}")
        print(f"{'='*72}")

        thread_text = build_thread_text(emails)

        # ── KROK 1 ────────────────────────────────────────────────────────────
        print(f"\n  [KROK 1 — voľné zhrnutie]", flush=True)
        summary = llm(PROMPT_STEP1.format(thread_text=thread_text), "krok1")
        print(summary)

        # ── KROK 2 ────────────────────────────────────────────────────────────
        print(f"\n  [KROK 2 — identifikácia]", flush=True)
        identification = llm(PROMPT_STEP2.format(summary=summary), "krok2")
        print(identification)

        print()

    conn.close()
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
