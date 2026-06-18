"""
Multi-model summary quality test.
Models: A) llama3.1:8b  B) llama3.3:70b  C) Mistral API  D) Claude Haiku

Run: python -m src._summary_test
     python -m src._summary_test --models A B       # subset
     python -m src._summary_test --models A B C D   # all (requires API keys)
"""

import argparse, io, os, re, sqlite3, sys, time
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

DB_PATH    = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

TEST_CIDS = [14125, 14129, 14050, 14052, 14154, 13753]

MAX_BODY = 1200
MAX_EP   = 20000

PROMPT = """\
Si asistent architektonickej firmy. Zhrň e-mailovú konverzáciu nižšie.

PRAVIDLÁ:
- Ak text obsahuje konkrétne čísla, rozmery, termíny alebo rozhodnutia — MUSÍŠ ich uviesť.
- Nezovšeobecňuj na prázdne frázy ako "riešia sa otázky" alebo "prebieha komunikácia".
- Ignoruj podpisy, telefónne čísla, adresy firiem a marketingové slogany v pätičkách.
- Píš po slovensky, vecne, v súvislých vetách.

Odpovedz VÝLUČNE v tomto formáte — PRESNE 2 riadky, žiadne odrážky, žiadne markdown:
ZHRNUTIE: <2-3 vety s faktami: čo sa rozhodlo, aké čísla, kto čo urobí>
OTVORENÉ: <body oddelené bodkočiarkou; alebo "—" ak nič>

=== KONVERZÁCIA ===
{thread_text}
=== KONIEC ==="""

# Each fact = list of alternative strings (any match → fact found).
# First string is the "label" shown in output. Covers SK/EN/abbrev variants.
CHECKS: dict[int, list[list[str]]] = {
    14125: [
        ["N2", "kategórie n2", "vozidlo n2", "nákladné n2"],
        ["bezbariér", "bezbariérový", "bezbariérovom", "bezbariérové", "bezbariérov"],
        ["3,8", "3.8"],
        ["1+1", "1 + 1", "koeficient 1"],
        ["vjazd", "vjazde", "vjazdov"],
    ],
    14129: [
        ["šachta", "šachty", "šachte", "šachtách"],
        ["denná", "denné", "dennú", "denného"],
    ],
    14050: [
        ["3m/s", "3 m/s", "3,0 m/s", "3.0 m/s", "rýchlosť 3", "metre za sekundu"],
        ["nevyšlo", "nestačí", "nestačilo", "nepostačuje", "nevychádzal", "nevyšla"],
        ["2 výťah", "dvoch výťah", "dva výťah", "2× výťah", "2 malé výťah"],
    ],
    14052: [
        ["74%", "74 %", "tau 74", "lt) 74", "74percent"],
        ["SVP", "svetlový výškový", "funkčná čiara", "funcna ciara", "svetlovýškový"],
        ["8.NP", "8. NP", "8NP", "8. nadpodlaž", "ôsmom nadpodlaž", "8np"],
        ["40.NP", "40. NP", "40NP", "40. nadpodlaž", "40np"],
        ["nevyhovuje", "nevyhovujú", "nevyhovel", "nespĺňa", "nespĺňajú", "nevyhov"],
        ["vyhovuje", "vyhovujú", "vyhovel", "spĺňa", "spĺňajú", "vyhovej"],
    ],
    14154: [
        ["178"],
        ["PM", "parkovacích miest", "parkovacie miesto", "parkovaci", "parkovanie miest"],
        ["garáž", "garáži", "garážach", "garážou"],
    ],
    13753: [
        ["205", "205m", "205 m"],
        ["930", "930m", "930 m²", "930m²"],
        ["54", "54 mm", "54mm"],
        ["175mm", "175 mm"],
        ["overrun", "prevýšenie", "prebeh", "prevyšnosť", "nadjazdnosť", "overrun"],
        ["machine room", "strojovňa", "strojový priestor", "strojovne", "strojovna", "strojovej", "strojnej", "strojná"],
    ],
}


def _fact_found(variants: list[str], combined: str) -> bool:
    return any(v.lower() in combined for v in variants)


def _fact_label(variants: list[str]) -> str:
    return variants[0]

# ── model definitions ─────────────────────────────────────────────────────────

MODELS = {
    "A": {"label": "llama3.1:8b   (local)",  "type": "ollama", "id": "llama3.1:8b"},
    "B": {"label": "llama3.3:70b  (local)",  "type": "ollama", "id": "llama3.3:70b"},
    "C": {"label": "mistral-small (API/EU)", "type": "mistral", "id": "mistral-small-latest"},
    "D": {"label": "claude-haiku  (API/US)", "type": "claude",  "id": "claude-haiku-4-5-20251001"},
}

# ── import pipeline helpers ───────────────────────────────────────────────────

from src.active_window import _active_streak, _latest_episode, _clean_body

# Old cleaner (line-level only, no inline quote truncation) for before/after comparison
import re as _re
_OLD_SIG = _re.compile(
    r"^(-{2,}|_{10,}|S pozdravom|Best regards|Regards,|Kind regards|"
    r"Sent from|Poslan[éo] z|This e-?mail|CONFIDENTIAL)", _re.IGNORECASE)
_OLD_FWDHDR = _re.compile(r"^(Od|From|Komu|To|Predmet|Subject|Dátum|Date|Sent|CC):\s",
                           _re.IGNORECASE)
_OLD_FWDSEP = _re.compile(r"^-{5,}\s*(Original|Forwarded|Pôvodná)", _re.IGNORECASE)

def _clean_body_old(text: str) -> str:
    lines, sig = [], False
    for line in (text or "").splitlines():
        s = line.strip()
        if not sig and (_OLD_SIG.match(s) or s in ("--", "— ")):
            sig = True
        if sig or s.startswith(">") or _OLD_FWDHDR.match(line) or _OLD_FWDSEP.match(s):
            continue
        lines.append(line)
    return _re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

# ── text builder ──────────────────────────────────────────────────────────────

def build_text(emails: list[dict], cleaner=_clean_body) -> str:
    parts = []
    for em in emails:
        dt   = (em["date"] or "")[:10]
        frm  = (em["from_address"] or "?").split("@")[0]
        body = cleaner(em.get("body_text") or "")
        if not body:
            continue
        if len(body) > MAX_BODY:
            body = body[:MAX_BODY] + "[...]"
        parts.append(f"[{dt}] {frm}\n{body}")
    text = "\n\n---\n\n".join(parts)
    return text[:MAX_EP] + "\n[OREZANÉ]" if len(text) > MAX_EP else text

# ── LLM backends ─────────────────────────────────────────────────────────────

def call_ollama(model_id: str, prompt: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": model_id, "prompt": prompt, "stream": False},
                          timeout=600)
        r.raise_for_status()
        data = r.json()
        elapsed = time.time() - t0
        return data.get("response", "").strip(), elapsed
    except Exception as e:
        return f"[CHYBA: {e}]", time.time() - t0


def call_mistral(model_id: str, prompt: str) -> tuple[str, float]:
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return "[CHÝBA MISTRAL_API_KEY]", 0.0
    t0 = time.time()
    try:
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_id, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        elapsed = time.time() - t0
        tok_in  = usage.get("prompt_tokens", 0)
        tok_out = usage.get("completion_tokens", 0)
        cost    = tok_in * 0.2e-6 + tok_out * 0.6e-6   # mistral-small pricing ($/token)
        return text, elapsed, tok_in, tok_out, cost
    except Exception as e:
        return f"[CHYBA: {e}]", time.time() - t0, 0, 0, 0.0


def call_claude(model_id: str, prompt: str) -> tuple[str, float]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[CHÝBA ANTHROPIC_API_KEY]", 0.0
    t0 = time.time()
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        text    = data["content"][0]["text"].strip()
        usage   = data.get("usage", {})
        elapsed = time.time() - t0
        tok_in  = usage.get("input_tokens", 0)
        tok_out = usage.get("output_tokens", 0)
        cost    = tok_in * 0.8e-6 + tok_out * 4e-6   # haiku-4-5 pricing ($/token)
        return text, elapsed, tok_in, tok_out, cost
    except Exception as e:
        return f"[CHYBA: {e}]", time.time() - t0, 0, 0, 0.0


def call_model(key: str, prompt: str):
    cfg = MODELS[key]
    if cfg["type"] == "ollama":
        result = call_ollama(cfg["id"], prompt)
        return result[0], result[1], 0, 0, 0.0
    if cfg["type"] == "mistral":
        r = call_mistral(cfg["id"], prompt)
        return r if len(r) == 5 else (r[0], r[1], 0, 0, 0.0)
    if cfg["type"] == "claude":
        r = call_claude(cfg["id"], prompt)
        return r if len(r) == 5 else (r[0], r[1], 0, 0, 0.0)
    return "[neznámy typ]", 0.0, 0, 0, 0.0

# ── output helpers ────────────────────────────────────────────────────────────

def parse(out: str) -> tuple[str, str]:
    m_s = re.search(r"ZHRNUTIE\s*:\s*(.+?)(?=OTVORENÉ\s*:|$)", out, re.DOTALL | re.IGNORECASE)
    m_o = re.search(r"OTVORENÉ\s*:\s*(.+)", out, re.DOTALL | re.IGNORECASE)
    return (m_s.group(1).strip() if m_s else out.strip()[:300],
            m_o.group(1).strip() if m_o else "—")

# ── main ─────────────────────────────────────────────────────────────────────

HARD_CIDS = [14052, 14125, 13753]   # CIDs where 8B was unstable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["A", "B"],
                        choices=list(MODELS), metavar="MODEL",
                        help="Models to test: A B C D")
    parser.add_argument("--consistency", action="store_true",
                        help="Run hard CIDs a 2nd time to verify stability")
    args = parser.parse_args()
    selected = args.models

    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    old_rows = {r["conversation_id"]: r
                for r in conn.execute("SELECT * FROM active_threads").fetchall()}

    # Pre-build episode texts (once, shared across models)
    episodes: dict[int, tuple[list, str, str, int, int]] = {}
    for cid in TEST_CIDS:
        full = [dict(r) for r in conn.execute(
            "SELECT id, date, subject, from_address, to_addresses, cc_addresses, body_text "
            "FROM emails WHERE conversation_id = ? ORDER BY date", (cid,)
        ).fetchall()]
        streak    = _active_streak(full)
        has_body  = [e for e in streak if (e.get("body_text") or "").strip()]
        ep        = _latest_episode(has_body)
        text_old  = build_text(ep["emails"], cleaner=_clean_body_old)
        text_new  = build_text(ep["emails"], cleaner=_clean_body)
        subj      = old_rows[cid]["subject"] if cid in old_rows else f"CID={cid}"
        episodes[cid] = (ep["emails"], text_new, subj, len(text_old), len(text_new))

    print("=" * 76)
    print(f"  MULTI-MODEL SUMMARY TEST  —  {', '.join(selected)}")
    print(f"  caps: MAX_BODY={MAX_BODY}  MAX_EP={MAX_EP}")
    print("=" * 76)

    # Results[model_key][cid] = (zhrnutie, otvorene, elapsed, tok_in, tok_out, cost)
    results: dict[str, dict[int, tuple]] = {k: {} for k in selected}
    total_costs: dict[str, float] = {k: 0.0 for k in selected}

    for key in selected:
        cfg = MODELS[key]
        print(f"\n{'━'*76}")
        print(f"  MODEL {key}: {cfg['label']}")
        print(f"{'━'*76}")

        for i, cid in enumerate(TEST_CIDS, 1):
            _, text, subj, n_old, n_new = episodes[cid]
            prompt = PROMPT.format(thread_text=text)

            reduction = (1 - n_new / n_old) * 100 if n_old else 0
            size_str  = f"{n_old:,}→{n_new:,} zn  (-{reduction:.0f}%)"
            print(f"  [{i}/6] CID={cid}  {subj[:40]}  {size_str}", end="", flush=True)
            print()
            raw, elapsed, tok_in, tok_out, cost = call_model(key, prompt)
            total_costs[key] += cost
            zhrn, otv = parse(raw)
            results[key][cid] = (zhrn, otv, elapsed, tok_in, tok_out, cost)

            checks   = CHECKS.get(cid, [])
            combined = (zhrn + " " + otv).lower()
            hits   = [v for v in checks if _fact_found(v, combined)]
            misses = [v for v in checks if not _fact_found(v, combined)]
            score  = f"{len(hits)}/{len(checks)}" if checks else "—"

            time_str = f"{elapsed:.0f}s" if elapsed >= 1 else f"{elapsed*1000:.0f}ms"
            print(f"       [{time_str}]  score={score}")
            print(f"     ZHRNUTIE: {zhrn[:220]}")
            if otv and otv != "—":
                print(f"     OTVORENÉ: {otv[:150]}")
            if misses:
                print(f"     CHÝBA: {', '.join(_fact_label(v) for v in misses)}")

    # ── Summary comparison table ─────────────────────────────────────────────
    print()
    print("=" * 76)
    print("  POROVNANIE — skóre (zachytené kľúčové fakty)")
    print("=" * 76)
    header = f"  {'CID':>6}  " + "  ".join(f"{k}:{MODELS[k]['id'].split(':')[0]:>12}" for k in selected)
    print(header)
    print("  " + "-" * 60)
    total_scores = {k: 0 for k in selected}
    total_checks = 0
    for cid in TEST_CIDS:
        checks = CHECKS.get(cid, [])
        total_checks += len(checks)
        row = f"  {cid:>6}  "
        for k in selected:
            zhrn, otv, *_ = results[k].get(cid, ("", "—", 0, 0, 0, 0.0))
            combined = (zhrn + " " + otv).lower()
            hits = sum(1 for v in checks if _fact_found(v, combined))
            total_scores[k] += hits
            row += f"  {hits}/{len(checks):>1}{'':>10}"
        print(row)
    print("  " + "-" * 60)
    totals = f"  {'SPOLU':>6}  "
    for k in selected:
        totals += f"  {total_scores[k]}/{total_checks}{'':>9}"
    print(totals)

    # ── Consistency check ─────────────────────────────────────────────────────
    if args.consistency:
        print()
        print("=" * 76)
        print("  KONZISTENTNOSŤ — 2. beh (hard CIDs: 14052, 14125, 13753)")
        print("=" * 76)
        for key in selected:
            print(f"\n  MODEL {key}: {MODELS[key]['label']}")
            for cid in HARD_CIDS:
                _, text, subj, _, _ = episodes[cid]
                raw2, e2, ti2, to2, c2 = call_model(key, PROMPT.format(thread_text=text))
                total_costs[key] += c2
                zhrn2, otv2 = parse(raw2)
                checks   = CHECKS.get(cid, [])
                combined2 = (zhrn2 + " " + otv2).lower()
                h2 = sum(1 for v in checks if _fact_found(v, combined2))
                zhrn1, otv1, *_ = results[key][cid]
                h1 = sum(1 for v in checks if _fact_found(v, (zhrn1 + " " + otv1).lower()))
                flag = "✓ KONZISTENTNÝ" if h1 == h2 else f"⚠ ROZDIEL  ({h1}→{h2})"
                print(f"     CID={cid}  beh1={h1}/{len(checks)}  beh2={h2}/{len(checks)}  {flag}")
                print(f"     beh2: {zhrn2[:180]}")

    # ── Cost / privacy table ─────────────────────────────────────────────────
    # Aggregate tokens from results
    tok_totals: dict[str, tuple[int, int]] = {}
    for k in selected:
        tin  = sum(results[k][c][3] for c in TEST_CIDS if c in results[k])
        tout = sum(results[k][c][4] for c in TEST_CIDS if c in results[k])
        tok_totals[k] = (tin, tout)

    CONVS_PER_DAY  = 130
    CONVS_PER_MONTH = CONVS_PER_DAY * 30

    print()
    print("  NÁKLADY A SÚKROMIE")
    print(f"  {'Model':22}  {'Súkr.':12}  {'Test (6)':>9}  {'/ konv':>7}  {'/ mesiac*':>10}  Poznámka")
    print("  " + "-" * 80)
    privacy = {"A": "100% lokálne", "B": "100% lokálne", "C": "GDPR/EÚ", "D": "US API"}
    notes   = {"A": "baseline",     "B": "~10min/run",   "C": "eu-central", "D": "rýchly"}
    for k in selected:
        tin, tout = tok_totals.get(k, (0, 0))
        test_cost  = total_costs[k]
        per_conv   = test_cost / len(TEST_CIDS) if test_cost > 0 else 0.0
        monthly    = per_conv * CONVS_PER_MONTH
        tok_str    = f"({tin//1000}k/{tout//1000}k tok)" if tin else ""
        if test_cost > 0:
            print(f"  {MODELS[k]['label']:22}  {privacy[k]:12}  ${test_cost:>7.4f}  "
                  f"${per_conv:>5.4f}  ${monthly:>8.2f}/mo  {tok_str}")
        else:
            print(f"  {MODELS[k]['label']:22}  {privacy[k]:12}  {'zadarmo':>9}  "
                  f"{'zadarmo':>7}  {'zadarmo':>10}  {notes[k]}")
    print(f"  * odhad: {CONVS_PER_DAY} konv/deň × 30 dní = {CONVS_PER_MONTH:,} konv/mesiac")

    print()
    print("=" * 76)
    print("  HOTOVO")
    print("=" * 76)
    conn.close()


if __name__ == "__main__":
    main()
