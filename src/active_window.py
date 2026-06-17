"""Active conversation window — "čo sa rieši" za posledných 30 dní.

Pipeline per aktívna konverzácia:
  1. Aktívny segment (streak)   — dozadu od okna, stop pri medzere >21d
  2. Hybridná segmentácia        — čas + téma pre bloky >TOPIC_MIN mailov
  3. LLM zhrnutie               — ZHRNUTIE + OTVORENÉ (1 call)
  4. Účastníci                  — deterministicky z domén, nie LLM

Ukladá do tabuľky active_threads.
Usage: python -m src.active_window [--dry-run]
"""

import io
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH         = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL       = "llama3.1:8b"

WINDOW_DAYS     = 30
MAX_STREAK_GAP  = 21   # days — stop tracing back at this gap
GAP_DAYS        = 6    # days — time-block boundary
TOPIC_MIN       = 20   # emails — last block larger than this → topic-segment
MAX_ID_SAMPLE   = 18   # emails sampled for topic identification
MAX_BODY        = 400  # chars per email in LLM input
MAX_ID_CHARS    = 6000 # chars for topic-ID prompt
MAX_EP_CHARS    = 8000 # chars for summary prompt
RAW_DESC_LIMIT  = 220  # parser: longer desc treated as raw email body
MAX_CONVS_LLM   = 25   # cap on LLM-processed conversations per run

PROMPT_LATEST = """\
Toto je e-mailová konverzácia z architektonickej firmy (najnovší aktívny úsek).
Odpovedz PRESNE v tomto formáte — 2 riadky, nič viac:

ZHRNUTIE: <2-3 vety: čo sa rieši, čo sa rozhodlo>
OTVORENÉ: <body oddelené bodkočiarkou; alebo "—" ak nič>

Píš po slovensky, vecne.

=== KONVERZÁCIA ===
{thread_text}
=== KONIEC ==="""

PROMPT_IDENTIFY = """\
E-maily z jedného bloku konverzácie (chronologicky). \
Identifikuj EPIZÓDY — tematické bloky s iným impulzom.

Pre každú epizódu vypíš PRESNE takto (1 riadok):
EPIZÓDA N: YYYY-MM-DD – YYYY-MM-DD | popis jednou vetou max 80 znakov

Popis nesmie byť citát. Nič iné nepiš.

=== EMAILY ===
{batch_text}
=== KONIEC ==="""

# ── known firms ────────────────────────────────────────────────────────────────

KNOWN_DOMAINS: dict[str, str] = {
    "gfi.sk":              "GFI",
    "jtre.sk":             "JTRE",
    "kcap.eu":             "KCAP",
    "ae7.com":             "AE7",
    "idealarch.com":       "Ideal Arch",
    "simplecode.sk":       "SimpleCode",
    "simulaciebudov.sk":   "Simulácie Budov",
    "promodel.sk":         "Promodel",
    "qubu.io":             "Qubu",
    "compass.sk":          "Compass",
    "pentarealestate.com": "Penta RE",
    "skgbc.org":           "SKGBC",
    "softhub.sk":          "SoftHub",
    "gehlpeople.com":      "Gehl",
    "ravago.com":          "Ravago",
    "atelier-edu.sk":      "Atelier EDU",
    "tristel.sk":          "Tristel",
    "ingsteel.sk":         "Ingsteel",
    "2create.sk":          "2create",
    "burohappold.com":     "BuroHappold",
    "smartcad.sk":         "SmartCAD",
    "milanilles.sk":       "Milan Illes AI",
    "mading.sk":           "Mading",
}

# ── noise detection ────────────────────────────────────────────────────────────

_NOISE_SUBJ = re.compile(
    r"(has \d+ updates? since|you have been mentioned|scheduler|"
    r"newsletter|unsubscribe|automatick[áa] správ|weekly report|"
    r"automatic notification)",
    re.IGNORECASE,
)
_NOISE_FROM = re.compile(
    r"(no-reply|noreply|donotreply|scheduler|newsletter|notification)",
    re.IGNORECASE,
)

def _is_noise(subject: str, from_addr: str) -> bool:
    return bool(_NOISE_SUBJ.search(subject or "") or _NOISE_FROM.search(from_addr or ""))

# ── text cleaning ──────────────────────────────────────────────────────────────

_SIG = re.compile(
    r"^(-{2,}|_{10,}|S pozdravom|Best regards|Regards,|Kind regards|"
    r"Sent from|Poslan[éo] z|This e-?mail|CONFIDENTIAL)",
    re.IGNORECASE,
)
_FWDHDR = re.compile(r"^(Od|From|Komu|To|Predmet|Subject|Dátum|Date|Sent|CC):\s",
                     re.IGNORECASE)
_FWDSEP = re.compile(r"^-{5,}\s*(Original|Forwarded|Pôvodná)", re.IGNORECASE)
_PREFIX = re.compile(
    r"^(Re|RE|Fwd|FW|Fw|Odp|Odp\.|VS|AW)\s*[:\s]\s*", re.IGNORECASE
)

def _clean_body(text: str) -> str:
    lines, sig = [], False
    for line in text.splitlines():
        s = line.strip()
        if not sig and (_SIG.match(s) or s in ("--", "— ")):
            sig = True
        if sig or s.startswith(">") or _FWDHDR.match(line) or _FWDSEP.match(s):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

def _norm_subj(s: str) -> str:
    s = (s or "").strip()
    while True:
        m = _PREFIX.match(s)
        if m:
            s = s[m.end():].strip()
        else:
            break
    return s

def _parse_dt(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat((s or "")[:19])
    except Exception:
        return None

# ── active streak ──────────────────────────────────────────────────────────────

def _active_streak(emails_asc: list[dict]) -> list[dict]:
    """Walk newest→oldest; stop at first gap > MAX_STREAK_GAP days."""
    if not emails_asc:
        return []
    rev    = list(reversed(emails_asc))
    streak = [rev[0]]
    for em in rev[1:]:
        d_new  = _parse_dt(streak[-1]["date"])
        d_prev = _parse_dt(em["date"])
        if d_new and d_prev and abs((d_new - d_prev).days) > MAX_STREAK_GAP:
            break
        streak.append(em)
    return sorted(streak, key=lambda x: x["date"] or "")

# ── participants (deterministic from domains) ──────────────────────────────────

_EMAIL_RE = re.compile(r"[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}")

def _participants_by_firm(emails: list[dict]) -> dict[str, list[str]]:
    fa: dict[str, set[str]] = defaultdict(set)
    for em in emails:
        for fld in (em.get("from_address"), em.get("to_addresses"), em.get("cc_addresses")):
            if not fld:
                continue
            for addr in _EMAIL_RE.findall(fld):
                addr   = addr.lower()
                domain = addr.split("@")[-1]
                firm   = KNOWN_DOMAINS.get(domain, f"ext:{domain}")
                fa[firm].add(addr.split("@")[0])
    return {f: sorted(v) for f, v in sorted(fa.items())}

def _fmt_participants(fm: dict[str, list[str]]) -> str:
    return " | ".join(f"{f}: {', '.join(ns)}" for f, ns in fm.items())

# ── thread text for LLM ────────────────────────────────────────────────────────

def _build_text(emails: list[dict], max_chars: int) -> str:
    parts = []
    for em in emails:
        dt   = (em["date"] or "")[:16]
        frm  = em.get("from_address") or "?"
        body = _clean_body(em.get("body_text") or "")
        if not body:
            continue
        if len(body) > MAX_BODY:
            body = body[:MAX_BODY] + "[...]"
        parts.append(f"[{dt}] {frm}\n{body}")
    text = "\n\n---\n\n".join(parts)
    return text[:max_chars] + "\n[...]" if len(text) > max_chars else text

# ── hybrid episode segmentation ────────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:T\d{2}:\d{2}(?::\d{2})?)?")

def _time_blocks(emails: list[dict]) -> list[list[dict]]:
    if not emails:
        return []
    segs = [[emails[0]]]
    for em in emails[1:]:
        prev = _parse_dt(segs[-1][-1]["date"])
        curr = _parse_dt(em["date"])
        if prev and curr and (curr - prev).days > GAP_DAYS:
            segs.append([])
        segs[-1].append(em)
    return segs

def _sample_emails(emails: list[dict], n: int = MAX_ID_SAMPLE) -> list[dict]:
    if len(emails) <= n:
        return emails
    c   = n // 3
    mid = len(emails) // 2 - c // 2
    pool = emails[:c] + emails[mid:mid + c] + emails[-c:]
    seen, out = set(), []
    for e in pool:
        if e["id"] not in seen:
            seen.add(e["id"])
            out.append(e)
    return sorted(out, key=lambda x: x["date"] or "")

def _parse_episodes(llm_out: str) -> list[tuple[str, str, str]]:
    """Tolerant parser: find any line with >=1 date → episode entry."""
    eps: list[tuple[str, str, str]] = []
    for raw in llm_out.splitlines():
        s = raw.strip()
        if not s:
            continue
        dates = _DATE_RE.findall(s)
        if not dates:
            if eps and len(s) > 8:
                clean = re.sub(r"^[\s\-–—|:.]+", "", s).strip()
                if clean and not re.match(r"(?i)^(epiz|=|koniec)", clean):
                    d_f, d_t, desc = eps[-1]
                    ext = desc + " / " + clean
                    if len(ext) <= RAW_DESC_LIMIT:
                        eps[-1] = (d_f, d_t, ext)
            continue
        d_from, d_to = dates[0], dates[-1]
        if d_from > d_to:                       # protection: swap inverted
            d_from, d_to = d_to, d_from
        s2 = re.sub(r"\[[\d\-T:]+\]", "", s)
        s2 = re.sub(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?", "", s2)
        s2 = re.sub(r"(?i)epiz[oó]da\s*n?\s*\d*\s*:?", "", s2)
        s2 = re.sub(r"^[\s\d.N\-–—|:()\[\]]+", "", s2)
        if "|" in s2:
            s2 = s2[s2.rfind("|") + 1:]
        desc = re.sub(r"^[\s\-–—|:.]+", "", s2).strip()
        if not desc or len(desc) > RAW_DESC_LIMIT:  # protection: raw body
            desc = "(bez popisu)"
        eps.append((d_from, d_to, desc))
    return eps

def _assign_emails(block: list[dict], eps: list[tuple]) -> list[tuple]:
    result = []
    for d_from, d_to, desc in eps:
        df = _parse_dt(d_from)
        dt = _parse_dt(d_to + "T23:59:59") if len(d_to) == 10 else _parse_dt(d_to)
        bucket = ([e for e in block if (ed := _parse_dt(e["date"])) and df <= ed <= dt]
                  if df and dt else [])
        result.append((d_from, d_to, desc, bucket))
    return result

def _latest_episode(streak: list[dict]) -> dict:
    """Hybrid segmentation → return latest episode {d_from, d_to, emails, segmented}."""
    blocks = _time_blocks(streak)
    last   = blocks[-1]

    if len(last) <= TOPIC_MIN:
        return {"d_from": last[0]["date"][:10], "d_to": last[-1]["date"][:10],
                "emails": last, "segmented": False}

    # Large last block — topic identification on sample
    sampled = _sample_emails(last)
    id_out  = _llm(PROMPT_IDENTIFY.format(batch_text=_build_text(sampled, MAX_ID_CHARS)))
    eps     = _parse_episodes(id_out)

    if not eps:
        return {"d_from": last[0]["date"][:10], "d_to": last[-1]["date"][:10],
                "emails": last, "segmented": True}

    assigned = _assign_emails(last, eps)
    for d_from, d_to, _, bucket in reversed(assigned):
        if bucket:
            return {"d_from": d_from, "d_to": d_to, "emails": bucket, "segmented": True}

    return {"d_from": last[0]["date"][:10], "d_to": last[-1]["date"][:10],
            "emails": last, "segmented": True}

# ── LLM ───────────────────────────────────────────────────────────────────────

def _llm(prompt: str) -> str:
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
                          timeout=180)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[CHYBA: {e}]"

def _llm_summarize(thread_text: str) -> tuple[str, str]:
    """Returns (summary, open_points) parsed from structured LLM output."""
    out    = _llm(PROMPT_LATEST.format(thread_text=thread_text))
    m_sum  = re.search(r"ZHRNUTIE\s*:\s*(.+?)(?=OTVORENÉ\s*:|$)", out,
                       re.DOTALL | re.IGNORECASE)
    m_open = re.search(r"OTVORENÉ\s*:\s*(.+)", out, re.DOTALL | re.IGNORECASE)
    return (m_sum.group(1).strip()  if m_sum  else out.strip(),
            m_open.group(1).strip() if m_open else "—")

# ── DB ─────────────────────────────────────────────────────────────────────────

def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS active_threads (
            conversation_id  INTEGER PRIMARY KEY,
            subject          TEXT,
            last_activity    TEXT,
            n_window         INTEGER,
            n_streak         INTEGER,
            episode_from     TEXT,
            episode_to       TEXT,
            summary          TEXT,
            open_points      TEXT,
            participants     TEXT,
            firms            TEXT,
            is_segmented     INTEGER,
            computed_at      TEXT
        )
    """)
    conn.execute("DELETE FROM active_threads")
    conn.commit()

def _db_save(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO active_threads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (row["conversation_id"], row["subject"], row["last_activity"],
          row["n_window"],        row["n_streak"],
          row["episode_from"],    row["episode_to"],
          row["summary"],         row["open_points"],
          row["participants"],    row["firms"],
          int(row["is_segmented"]), row["computed_at"]))
    conn.commit()

# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # ── KROK A — okno ─────────────────────────────────────────────────────────
    max_date     = conn.execute("SELECT MAX(date) FROM emails").fetchone()[0]
    window_end   = datetime.fromisoformat(max_date[:19])
    window_start = (window_end - timedelta(days=WINDOW_DAYS)).isoformat()[:10]

    n_em, n_cv = conn.execute("""
        SELECT COUNT(*), COUNT(DISTINCT conversation_id)
        FROM emails WHERE date >= ? AND conversation_id IS NOT NULL
    """, (window_start,)).fetchone()

    print(f"=== KROK A — okno {window_start} – {max_date[:10]} ===")
    print(f"  Emailov      : {n_em:,}")
    print(f"  Konverzácií  : {n_cv:,}")
    print()

    # ── KROK B — streak (bez LLM) ─────────────────────────────────────────────
    conv_rows = conn.execute("""
        SELECT conversation_id, MAX(date) as last
        FROM emails WHERE date >= ? AND conversation_id IS NOT NULL
        GROUP BY conversation_id ORDER BY last DESC
    """, (window_start,)).fetchall()

    print(f"=== KROK B — streak + epizóda ({len(conv_rows)} konverzácií, bez LLM) ===")

    n_noise = n_sing = 0
    candidates: list[dict] = []

    for row in conv_rows:
        cid = row["conversation_id"]
        # Lightweight load (no body) for streak/noise check
        light = [dict(r) for r in conn.execute("""
            SELECT id, date, subject, from_address
            FROM emails WHERE conversation_id = ? ORDER BY date
        """, (cid,)).fetchall()]
        if not light:
            continue

        newest = light[-1]
        if _is_noise(newest.get("subject") or "", newest.get("from_address") or ""):
            n_noise += 1
            continue

        streak = _active_streak(light)
        if len(streak) < 2:
            n_sing += 1
            continue

        n_window    = sum(1 for e in light if (e.get("date") or "") >= window_start)
        blocks      = _time_blocks(streak)
        needs_topic = len(blocks[-1]) > TOPIC_MIN

        candidates.append({
            "conversation_id": cid,
            "subject":         _norm_subj(newest.get("subject") or ""),
            "last_activity":   streak[-1]["date"][:10],
            "n_window":        n_window,
            "n_streak":        len(streak),
            "needs_topic":     needs_topic,
        })

    n_large = sum(1 for c in candidates if c["needs_topic"])
    n_small = len(candidates) - n_large
    to_llm  = candidates[:MAX_CONVS_LLM]

    print(f"  Noise vynechané          : {n_noise}")
    print(f"  Singletony (<2 mailov)   : {n_sing}")
    print(f"  Aktívnych konverzácií    : {len(candidates)}")
    print(f"    krátke (≤{TOPIC_MIN}m) : {n_small}")
    print(f"    dlhé  (>{TOPIC_MIN}m)  : {n_large}  → téma segmentácia")
    print(f"  Na LLM: {len(to_llm)}"
          + (f"  (preskočených {len(candidates)-len(to_llm)})" if len(candidates) > MAX_CONVS_LLM else ""))
    print()

    # ── KROK C — LLM + uloženie ───────────────────────────────────────────────
    if not dry_run:
        _ensure_table(conn)

    computed_at = datetime.now().isoformat()[:19]
    results: list[dict] = []

    print(f"=== KROK C — LLM ({len(to_llm)} konverzácií) ===")
    print()

    for i, c in enumerate(to_llm, 1):
        cid = c["conversation_id"]
        print(f"  [{i:02d}/{len(to_llm)}] {c['subject'][:55]}  [{c['last_activity']}]",
              flush=True)

        # Full load (body + addresses) for this conversation
        full = [dict(r) for r in conn.execute("""
            SELECT id, date, subject, from_address, to_addresses, cc_addresses, body_text
            FROM emails WHERE conversation_id = ? ORDER BY date
        """, (cid,)).fetchall()]

        streak   = _active_streak(full)
        has_body = [e for e in streak if (e.get("body_text") or "").strip()]

        if not has_body:
            print("  → (žiadne telá mailov — preskočené)\n")
            continue

        ep = _latest_episode(has_body)
        if not ep["emails"]:
            print("  → (prázdna epizóda — preskočené)\n")
            continue

        firm_map          = _participants_by_firm(streak)
        ep_text           = _build_text(ep["emails"], MAX_EP_CHARS)
        summary, open_pts = _llm_summarize(ep_text)

        print(f"  → {summary[:160]}")
        if open_pts and open_pts != "—":
            print(f"     Otvorené: {open_pts[:120]}")
        print()

        row = {
            "conversation_id": cid,
            "subject":         c["subject"],
            "last_activity":   c["last_activity"],
            "n_window":        c["n_window"],
            "n_streak":        c["n_streak"],
            "episode_from":    ep["d_from"],
            "episode_to":      ep["d_to"],
            "summary":         summary,
            "open_points":     open_pts,
            "participants":    _fmt_participants(firm_map),
            "firms":           "|".join(sorted(firm_map)),
            "is_segmented":    ep["segmented"],
            "computed_at":     computed_at,
        }
        if not dry_run:
            _db_save(conn, row)
        results.append(row)

    # ── Výsledný sumár ────────────────────────────────────────────────────────
    n_seg   = sum(1 for r in results if r["is_segmented"])
    n_short = len(results) - n_seg

    print()
    print("=" * 72)
    print(f"  ČO SA RIEŠI — {window_start} – {max_date[:10]}")
    print(f"  {len(results)} konverzácií  |  {n_seg} segmentovaných  |  {n_short} krátkych")
    print("=" * 72)
    print()

    for r in results:
        seg_tag  = " [SEG]" if r["is_segmented"] else ""
        ep_range = f"{r['episode_from']} – {r['episode_to']}"
        print(f"{'─'*72}")
        print(f"  {r['subject'][:60]}{seg_tag}")
        print(f"  {r['last_activity']}  |  okno:{r['n_window']}m  "
              f"streak:{r['n_streak']}m  |  ep: {ep_range}")
        print(f"  {r['participants'][:130]}")
        print(f"{'─'*72}")
        print(f"  {r['summary']}")
        if r["open_points"] and r["open_points"] != "—":
            print(f"  Otvorené: {r['open_points']}")
        print()

    if not dry_run:
        print(f"  Uložené do active_threads: {len(results)} riadkov  [{computed_at}]")
    conn.close()
    print("=== DONE ===")


if __name__ == "__main__":
    main()
