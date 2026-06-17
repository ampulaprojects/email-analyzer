"""Hybrid episode segmentation: time skeleton + LLM topics on large blocks.

Test on conv_id=12634 (AI f1, 109 emails).
Usage: python -m src._episode_test
"""

import io
import os
import re
import sqlite3
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH        = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL      = "llama3.1:8b"

CONV_ID        = 12634
GAP_DAYS       = 6    # time gap → new block
TOPIC_MIN      = 20   # blocks with > TOPIC_MIN emails → topic-segment inside
MAX_ID_SAMPLE  = 18   # emails sampled from large block for identification
MAX_BODY       = 350  # chars per email body
MAX_SEG_CHARS  = 6000 # max chars sent to LLM for identification
MAX_SUM_CHARS  = 5000 # max chars sent to LLM for summarization
RAW_DESC_LIMIT = 220  # if desc > this chars, treat as raw email text

# ── cleaning ──────────────────────────────────────────────────────────────────

_SIG = re.compile(
    r"^(-{2,}|_{10,}|S pozdravom|Best regards|Regards,|Kind regards|"
    r"Sent from|Poslan[éo] z|This e-?mail|CONFIDENTIAL)",
    re.IGNORECASE,
)
_FWDHDR = re.compile(r"^(Od|From|Komu|To|Predmet|Subject|Dátum|Date|Sent|CC):\s",
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


def _parse_dt(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat((s or "")[:19])
    except Exception:
        return None


def _fmt_email(em: dict) -> str | None:
    dt   = (em["date"] or "")[:10]
    frm  = (em["from_address"] or "?").split("@")[0]
    body = clean_body(em["body_text"] or "")
    if not body:
        return None
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY] + "[...]"
    return f"[{dt}] {frm}: {body}"


def _build_text(emails: list[dict], max_chars: int) -> str:
    parts = [p for e in emails if (p := _fmt_email(e))]
    text  = "\n\n---\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... skrátené ...]"
    return text


# ── LLM ──────────────────────────────────────────────────────────────────────

PROMPT_IDENTIFY = """\
Toto sú e-maily z jedného bloku e-mailovej konverzácie (chronologicky). \
Identifikuj EPIZÓDY — tematické bloky kde sa riešila iná vec alebo nastal nový impulz.

Pre každú epizódu vypíš PRESNE v tomto formáte (každá = 1 riadok):
EPIZÓDA N: YYYY-MM-DD – YYYY-MM-DD | popis jednou vetou (max 80 znakov)

N je číslo, dátumy sú z emailov, popis je čo sa riešilo. \
Popis nesmie byť priamy citát z emailu. Nič iné nepiš.

=== EMAILY ===
{batch_text}
=== KONIEC ==="""

PROMPT_SUMMARIZE = """\
Zhrň nasledujúci úsek e-mailovej konverzácie v 2-3 vetách. \
Čo sa riešilo, čo sa rozhodlo, čo ostalo otvorené. Píš po slovensky, vecne.

=== ÚSEK ===
{seg_text}
=== KONIEC ==="""


def llm(prompt: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[CHYBA: {e}]"


# ── tolerant parser ───────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:T\d{2}:\d{2}(?::\d{2})?)?")


def _clean_desc(line: str) -> str:
    s = re.sub(r"\[[\d\-T:]+\]", "", line)
    s = re.sub(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?", "", s)
    s = re.sub(r"(?i)epiz[oó]da\s*n?\s*\d*\s*:?", "", s)
    s = re.sub(r"^[\s\d.N\-–—|:()\[\]]+", "", s)
    if "|" in s:
        s = s[s.rfind("|") + 1:]
    return re.sub(r"^[\s\-–—|:.]+", "", s).strip()


def parse_episodes(llm_out: str) -> list[tuple[str, str, str]]:
    episodes: list[tuple[str, str, str]] = []
    for raw in llm_out.splitlines():
        s = raw.strip()
        if not s:
            continue
        dates = _DATE_RE.findall(s)
        if dates:
            d_from = dates[0]
            d_to   = dates[-1]
            # Protection: swap inverted dates
            if d_from > d_to:
                d_from, d_to = d_to, d_from
            desc = _clean_desc(s)
            # Protection: raw email body masquerading as description
            if not desc or len(desc) > RAW_DESC_LIMIT:
                desc = "(bez popisu)"
            episodes.append((d_from, d_to, desc))
        elif episodes and len(s) > 8:
            cleaned = re.sub(r"^[\s\-–—|:.]+", "", s).strip()
            if cleaned and not re.match(r"(?i)^(epiz|koniec|=)", cleaned):
                d_f, d_t, desc = episodes[-1]
                extended = desc + " / " + cleaned
                if len(extended) <= RAW_DESC_LIMIT:
                    episodes[-1] = (d_f, d_t, extended)
    return episodes


# ── time segmentation ─────────────────────────────────────────────────────────

def time_segments(emails: list[dict], gap_days: int = GAP_DAYS) -> list[list[dict]]:
    if not emails:
        return []
    segs = [[emails[0]]]
    for em in emails[1:]:
        prev_dt = _parse_dt(segs[-1][-1]["date"])
        curr_dt = _parse_dt(em["date"])
        if prev_dt and curr_dt and (curr_dt - prev_dt).days > gap_days:
            segs.append([])
        segs[-1].append(em)
    return segs


def _sample_for_id(emails: list[dict], n: int = MAX_ID_SAMPLE) -> list[dict]:
    if len(emails) <= n:
        return emails
    chunk = n // 3
    mid_s = len(emails) // 2 - chunk // 2
    sampled = emails[:chunk] + emails[mid_s:mid_s + chunk] + emails[-chunk:]
    seen, result = set(), []
    for e in sampled:
        if e["id"] not in seen:
            seen.add(e["id"])
            result.append(e)
    return sorted(result, key=lambda x: x["date"] or "")


def assign_emails(
    block: list[dict], episodes: list[tuple[str, str, str]]
) -> list[tuple[str, str, str, list[dict]]]:
    result = []
    for d_from, d_to, desc in episodes:
        df = _parse_dt(d_from)
        dt = _parse_dt(d_to + "T23:59:59") if len(d_to) == 10 else _parse_dt(d_to)
        bucket = ([e for e in block if (ed := _parse_dt(e["date"])) and df <= ed <= dt]
                  if df and dt else [])
        result.append((d_from, d_to, desc, bucket))
    return result


def check_overlaps(episodes: list[dict]) -> list[tuple[int, int]]:
    def to_dt(s: str) -> datetime | None:
        return _parse_dt(s + "T23:59:59") if len(s) == 10 else _parse_dt(s)
    pairs = []
    for i in range(len(episodes)):
        df1 = _parse_dt(episodes[i]["d_from"])
        dt1 = to_dt(episodes[i]["d_to"])
        for j in range(i + 1, len(episodes)):
            df2 = _parse_dt(episodes[j]["d_from"])
            dt2 = to_dt(episodes[j]["d_to"])
            if df1 and dt1 and df2 and dt2 and df1 <= dt2 and df2 <= dt1:
                pairs.append((i + 1, j + 1))
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, date, subject, from_address, body_text
        FROM emails WHERE conversation_id = ?
        ORDER BY date
    """, (CONV_ID,)).fetchall()
    emails    = [dict(r) for r in rows]
    has_body  = [e for e in emails if (e.get("body_text") or "").strip()]

    print(f"conv_id={CONV_ID}  {len(emails)} mailov ({len(has_body)} s telom)")
    print(f"rozsah: {emails[0]['date'][:10]} – {emails[-1]['date'][:10]}")
    print()

    # ── Krok 1: časová kostra ─────────────────────────────────────────────────
    time_blocks = time_segments(has_body)
    print(f"=== KROK 1 — ČASOVÁ KOSTRA ({len(time_blocks)} blokov, medzera >{GAP_DAYS}d) ===")
    print()
    for i, blk in enumerate(time_blocks, 1):
        action = "→ témová segmentácia (vnútri bloku)" if len(blk) > TOPIC_MIN else "→ 1 epizóda"
        print(f"  Blok {i:02d}  {blk[0]['date'][:10]} – {blk[-1]['date'][:10]}"
              f"  ({len(blk):3d} mailov)  {action}")
    print()

    # ── Krok 2: hybridná segmentácia ─────────────────────────────────────────
    print(f"=== KROK 2 — HYBRIDNÁ SEGMENTÁCIA ===")
    print()

    final_eps: list[dict] = []

    for blk_idx, block in enumerate(time_blocks, 1):
        blk_d0 = block[0]["date"][:10]
        blk_d1 = block[-1]["date"][:10]

        if len(block) <= TOPIC_MIN:
            print(f"  Blok {blk_idx:02d} [{blk_d0}–{blk_d1}] ({len(block)} mailov)"
                  f" → 1 epizóda...", flush=True)
            seg_text = _build_text(block, MAX_SUM_CHARS)
            summary  = llm(PROMPT_SUMMARIZE.format(seg_text=seg_text))
            final_eps.append({
                "block": blk_idx, "d_from": blk_d0, "d_to": blk_d1,
                "n_emails": len(block), "desc": None,
                "summary": summary, "from_topic": False,
            })
            print(f"  → {summary[:220]}")
            print()

        else:
            print(f"  Blok {blk_idx:02d} [{blk_d0}–{blk_d1}] ({len(block)} mailov)"
                  f" → témová ID (vzorka {MAX_ID_SAMPLE})...", flush=True)
            sampled  = _sample_for_id(block)
            id_out   = llm(PROMPT_IDENTIFY.format(batch_text=_build_text(sampled, MAX_SEG_CHARS)))
            episodes = parse_episodes(id_out)

            print(f"  LLM ID výstup ({len(id_out)} znakov) → {len(episodes)} epizód")
            for ep in episodes:
                print(f"    [{ep[0]} – {ep[1]}] {ep[2][:70]}")

            if not episodes:
                print(f"  [!] Žiadne epizódy → fallback: celý blok = 1 epizóda")
                seg_text = _build_text(block, MAX_SUM_CHARS)
                summary  = llm(PROMPT_SUMMARIZE.format(seg_text=seg_text))
                final_eps.append({
                    "block": blk_idx, "d_from": blk_d0, "d_to": blk_d1,
                    "n_emails": len(block), "desc": None,
                    "summary": summary, "from_topic": False,
                })
            else:
                assigned = assign_emails(block, episodes)
                for d_from, d_to, desc, bucket in assigned:
                    print(f"  → Ep [{d_from}–{d_to}] ({len(bucket)} mailov) zhrnutie...",
                          flush=True)
                    summary = (llm(PROMPT_SUMMARIZE.format(
                                   seg_text=_build_text(bucket, MAX_SUM_CHARS)))
                               if bucket else "(žiadne maily v rozsahu)")
                    final_eps.append({
                        "block": blk_idx, "d_from": d_from, "d_to": d_to,
                        "n_emails": len(bucket), "desc": desc,
                        "summary": summary, "from_topic": True,
                    })
                    print(f"     {summary[:200]}")
            print()

    # ── Výsledná tabuľka ──────────────────────────────────────────────────────

    print("=" * 72)
    print(f"  VÝSLEDOK: {len(final_eps)} epizód celkovo")
    print("=" * 72)
    print()
    print(f"  {'#':>3}  {'bl':>2}  {'dátum od':>10}  {'dátum do':>10}  "
          f"{'mai':>4}  T  popis / zhrnutie (skrátené)")
    print("  " + "-" * 78)
    for i, ep in enumerate(final_eps, 1):
        typ    = "T" if ep["from_topic"] else "C"
        marker = " ◄" if i == len(final_eps) else ""
        desc   = (ep["desc"] or ep["summary"] or "")
        # Show first sentence of summary or desc
        first_sent = re.split(r"[.!?]\s", desc)[0][:60]
        print(f"  {i:>3}  {ep['block']:>2}  {ep['d_from']:>10}  {ep['d_to']:>10}  "
              f"{ep['n_emails']:>4}  {typ}  {first_sent}{marker}")
    print()
    print("  T=téma (LLM vnútri bloku)  C=čas (celý blok)")
    print()

    # ── Najnovšia epizóda (plné zhrnutie) ────────────────────────────────────

    last = final_eps[-1]
    print("=" * 72)
    print("  *** NAJNOVŠIA EPIZÓDA — do denného sumáru ***")
    print("=" * 72)
    print(f"  Epizóda {len(final_eps)}  Blok {last['block']}  "
          f"{last['d_from']} – {last['d_to']}  ({last['n_emails']} mailov)")
    if last["desc"]:
        print(f"  Téma (ID): {last['desc']}")
    print()
    print(last["summary"])
    print()

    # ── Kontrola kvality ──────────────────────────────────────────────────────

    print("=" * 72)
    print("  KONTROLA KVALITY")
    print("=" * 72)

    overlaps = check_overlaps(final_eps)
    print(f"  Prekryvy dátumov     : {len(overlaps)}", end="")
    if overlaps:
        for i, j in overlaps[:5]:
            print(f"\n    Ep.{i} [{final_eps[i-1]['d_from']}–{final_eps[i-1]['d_to']}]"
                  f" × Ep.{j} [{final_eps[j-1]['d_from']}–{final_eps[j-1]['d_to']}]",
                  end="")
    print()

    inverted = [i+1 for i, ep in enumerate(final_eps) if ep["d_from"] > ep["d_to"]]
    print(f"  Invertované dátumy   : {len(inverted)}"
          + (f"  → ep {inverted}" if inverted else "  ✓"))

    raw_desc = [i+1 for i, ep in enumerate(final_eps)
                if ep.get("desc") and len(ep["desc"]) > RAW_DESC_LIMIT]
    print(f"  Raw telá v popise    : {len(raw_desc)}"
          + (f"  → ep {raw_desc}" if raw_desc else "  ✓"))

    empty = [i+1 for i, ep in enumerate(final_eps) if ep["n_emails"] == 0]
    print(f"  Epizódy bez mailov   : {len(empty)}"
          + (f"  → ep {empty}" if empty else "  ✓"))

    last_date = max(ep["d_to"] for ep in final_eps)
    ok = last_date >= "2026-06-15"
    print(f"  Posledný dátum       : {last_date}"
          + ("  ✓ (2026-06-15 pokryté)" if ok else "  [!] 2026-06-15 CHÝBA"))

    print()
    conn.close()
    print("=== DONE ===")


if __name__ == "__main__":
    main()
