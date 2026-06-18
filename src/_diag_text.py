"""
Diagnostic: are key facts actually present in the text sent to the model?
Run: python -m src._diag_text
"""
import io, os, re, sqlite3, sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

DB_PATH = os.environ.get("DB_PATH", "data/emails.db")
MAX_BODY = 1200
MAX_EP   = 20000

from src.active_window import _active_streak, _latest_episode, _clean_body

def build_text(emails: list[dict]) -> str:
    parts = []
    for em in emails:
        dt   = (em["date"] or "")[:10]
        frm  = (em["from_address"] or "?").split("@")[0]
        body = _clean_body(em.get("body_text") or "")
        if not body:
            continue
        if len(body) > MAX_BODY:
            body = body[:MAX_BODY] + "[...]"
        parts.append(f"[{dt}] {frm}\n{body}")
    text = "\n\n---\n\n".join(parts)
    return text[:MAX_EP] + "\n[OREZANÉ]" if len(text) > MAX_EP else text

# Facts to search — variants for each check token
CHECKS_DETAIL: dict[int, list[tuple[str, list[str]]]] = {
    14050: [
        ("3m/s",    ["3m/s", "3 m/s", "3,0 m/s", "3.0 m/s", "metre za sekundu", "m/s"]),
        ("nevyšlo", ["nevyšlo", "nevychádza", "nestačí", "nestaci", "nevyhovel", "nevyhov",
                     "nezodpoved", "nepostačuje", "nepostaci"]),
        ("2 výťah", ["2 výťah", "2 vytah", "dva výťah", "dva vytah", "dvoch výťah",
                     "2× výťah", "2x výťah"]),
    ],
    14052: [
        ("74%",     ["74%", "74 %", "74percent"]),
        ("SVP",     ["SVP", "svetelné výpočty", "svetlotechni"]),
        ("8.NP",    ["8.NP", "8. NP", "8NP", "8. nadpodlaž", "8.nadpodlaž"]),
        ("40.NP",   ["40.NP", "40. NP", "40NP", "40. nadpodlaž"]),
        ("vyhovuje",["vyhovuje", "vyhovujú", "nevyhovuje", "nevyhovujú", "vyhovela"]),
    ],
    14125: [
        ("N2",      ["N2", "úroveň N2", "podlažie N2", "N-2"]),
        ("3,8",     ["3,8", "3.8", "3,80", "3.80"]),
        ("bezbariér",["bezbariér", "bezbariérov", "bezbarierový", "bezbariérový",
                      "wheelchair", "invalidný", "ZŤP"]),
        ("1+1",     ["1+1", "1 + 1"]),
        ("vjazd",   ["vjazd", "vjazdov", "vchod", "vstup do garáže", "nájazd"]),
    ],
}

CIDS = [14050, 14052, 14125]

def find_in_lines(text: str, variants: list[str]) -> list[tuple[int, str, str]]:
    """Return list of (line_num, matched_variant, line_content) for each hit."""
    hits = []
    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        for v in variants:
            if v.lower() in line.lower():
                hits.append((i, v, line.strip()))
                break  # one hit per line
    return hits

def main() -> None:
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    for cid in CIDS:
        full = [dict(r) for r in conn.execute(
            "SELECT id, date, subject, from_address, body_text "
            "FROM emails WHERE conversation_id = ? ORDER BY date", (cid,)
        ).fetchall()]
        streak   = _active_streak(full)
        has_body = [e for e in streak if (e.get("body_text") or "").strip()]
        ep       = _latest_episode(has_body)
        text     = build_text(ep["emails"])

        subj = full[0]["subject"] if full else f"CID={cid}"
        n_emails = len(ep["emails"])

        print()
        print("=" * 80)
        print(f"  CID={cid}  |  {n_emails} mailov v epizóde  |  {len(text):,} znakov")
        print(f"  {subj}")
        print("=" * 80)

        # Fact search
        checks = CHECKS_DETAIL.get(cid, [])
        fact_results = []
        for token, variants in checks:
            hits = find_in_lines(text, variants)
            fact_results.append((token, variants, hits))

        print()
        print("  ── ANALÝZA FAKTOV ──────────────────────────────────────────────────────────")
        counts = {"A": 0, "B": 0, "C": 0}
        for token, variants, hits in fact_results:
            if hits:
                # Fact IS in text — model had it
                print(f"  [A] '{token}' — NÁJDENÝ v texte ({len(hits)} výskyt/ov)")
                for lineno, matched_v, linecontent in hits[:3]:
                    print(f"       riadok {lineno:>4}: '{matched_v}'  →  {linecontent[:90]}")
                counts["A"] += 1
            else:
                # Fact NOT in text
                print(f"  [B] '{token}' — NIE je v texte  (hľadané: {', '.join(variants[:4])})")
                counts["B"] += 1

        # CHECKS token match check (literal match — might be too strict)
        print()
        print("  ── CHECKS PRÍSNOSŤ (presný reťazec) ──────────────────────────────────────")
        from src._summary_test import CHECKS
        check_tokens = CHECKS.get(cid, [])
        for tok in check_tokens:
            in_text = tok.lower() in text.lower()
            label = "✓ v texte" if in_text else "✗ NIE v texte"
            print(f"       CHECKS['{tok}'] → {label}")

        print()
        print(f"  ── ZÁVER: A={counts['A']} (v texte, model ignoroval)  "
              f"B={counts['B']} (nie v texte = cap/cleaner)  ──")

        # Full text dump
        print()
        print("  ── PLNÝ TEXT MODELU (" + "─" * 45 + ")")
        print()
        for lineno, line in enumerate(text.split("\n"), 1):
            print(f"  {lineno:>4} │ {line}")

        print()

    conn.close()

if __name__ == "__main__":
    main()
