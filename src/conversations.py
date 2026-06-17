"""Conversation grouping: header chains + normalized subject + participant/time guardrail.

Algorithm (3 phases):
  1. Header chains — In-Reply-To + References (primary, trusted)
  2. Normalized subject — reconnects broken chains via subject matching
  3. Guardrail — blocks subject merges where gap > 60 days AND no shared participant

Output: conversation_id column in emails (does NOT overwrite thread_id).

Usage: python -m src.conversations
"""

import io
import os
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

DB_PATH              = os.environ.get("DB_PATH", "data/emails.db")
MAX_SUBJ_GAP_DAYS    = 60   # guardrail: max days for subject-only merge


# ── helpers ───────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


_PREFIX_RE = re.compile(
    r"^(Re|RE|Fwd|FW|Fw|Odp|Odp\.|Odpoveď|VS|AW|Pfwd)\s*[:\s]\s*",
    re.IGNORECASE,
)


def normalize_subject(s: str) -> str:
    s = (s or "").strip()
    while True:
        m = _PREFIX_RE.match(s)
        if m:
            s = s[m.end():].strip()
        else:
            break
    # remove diacritics
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.lower()).strip()


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str[:19])   # YYYY-MM-DDTHH:MM:SS
    except Exception:
        return None


_EMAIL_RE = re.compile(r"[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}")


def _participants(row: dict) -> frozenset:
    addrs = set()
    for field in (row["from_address"], row["to_addresses"], row["cc_addresses"]):
        if field:
            addrs.update(m.lower() for m in _EMAIL_RE.findall(field))
    return frozenset(addrs)


def _msgids_from_refs(refs: str | None) -> list[str]:
    if not refs:
        return []
    return re.findall(r"<[^>]+>", refs)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_column(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(emails)").fetchall()}
    if "conversation_id" not in cols:
        conn.execute("ALTER TABLE emails ADD COLUMN conversation_id INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_emails_conv ON emails(conversation_id)"
        )
        conn.commit()
        print("  Stĺpec conversation_id pridaný.", flush=True)
    else:
        print("  Stĺpec conversation_id už existuje.", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=== conversations.py ===\n")
    _ensure_column(conn)

    # ── load emails ───────────────────────────────────────────────────────────
    print("Nacitavam emaily...", flush=True)
    rows = conn.execute("""
        SELECT id, message_id, thread_id, in_reply_to, "references", date,
               from_address, to_addresses, cc_addresses, subject
        FROM emails ORDER BY date
    """).fetchall()
    emails  = [dict(r) for r in rows]
    n       = len(emails)
    idx_map = {em["id"]: i for i, em in enumerate(emails)}   # db_id → list index
    print(f"  {n:,} emailov\n", flush=True)

    # ── precompute per-email derived data ─────────────────────────────────────
    msgid_to_i: dict[str, int] = {}
    for i, em in enumerate(emails):
        mid = (em["message_id"] or "").strip()
        if mid:
            msgid_to_i[mid] = i

    parts   = [_participants(em)              for em in emails]
    dates   = [_parse_date(em["date"])        for em in emails]
    nsubjs  = [normalize_subject(em["subject"] or "") for em in emails]

    # ── phase 1: header chains ────────────────────────────────────────────────
    print("=== Faza 1: Hlavičkové reťaze (In-Reply-To + References) ===", flush=True)
    uf = UnionFind(n)
    header_merges = 0

    for i, em in enumerate(emails):
        # in_reply_to
        irt = (em["in_reply_to"] or "").strip()
        if irt and irt in msgid_to_i:
            if uf.union(i, msgid_to_i[irt]):
                header_merges += 1

        # references
        for ref in _msgids_from_refs(em["references"]):
            if ref in msgid_to_i:
                if uf.union(i, msgid_to_i[ref]):
                    header_merges += 1

    comps_p1 = len({uf.find(i) for i in range(n)})
    print(f"  {header_merges:,} header-spojení  →  {comps_p1:,} konverzácií\n", flush=True)

    # ── phase 2: subject + guardrail ─────────────────────────────────────────
    print("=== Faza 2: Normalizovaný subject + poistka ===", flush=True)
    subj_groups: dict[str, list[int]] = defaultdict(list)
    for i, ns in enumerate(nsubjs):
        if ns:
            subj_groups[ns].append(i)

    subj_merges   = 0
    subj_blocked  = 0

    for ns, idxs in subj_groups.items():
        if len(idxs) < 2:
            continue
        # sort by date (None → epoch)
        idxs_sorted = sorted(idxs, key=lambda k: dates[k] or datetime.min)

        for a, b in zip(idxs_sorted, idxs_sorted[1:]):
            if uf.find(a) == uf.find(b):
                continue

            # time gap
            da, db  = dates[a], dates[b]
            gap_ok  = True
            gap_days = None
            if da and db:
                gap_days = abs((db - da).days)
                gap_ok   = gap_days <= MAX_SUBJ_GAP_DAYS

            # participant overlap
            shared = bool(parts[a] & parts[b])

            if gap_ok or shared:
                uf.union(a, b)
                subj_merges += 1
            else:
                subj_blocked += 1

    comps_p2 = len({uf.find(i) for i in range(n)})
    print(f"  {subj_merges:,} subject-spojení")
    print(f"  {subj_blocked:,} zablokovaných (medzera >{MAX_SUBJ_GAP_DAYS}d + žiadni spoloční účastníci)")
    print(f"  →  {comps_p2:,} konverzácií\n", flush=True)

    # ── assign sequential conversation_ids ────────────────────────────────────
    root_to_cid: dict[int, int] = {}
    cid_counter = 0
    conv_ids: list[int] = []
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_cid:
            cid_counter += 1
            root_to_cid[r] = cid_counter
        conv_ids.append(root_to_cid[r])

    # ── write to DB ───────────────────────────────────────────────────────────
    print("Zapisujem conversation_id...", flush=True)
    conn.executemany(
        "UPDATE emails SET conversation_id = ? WHERE id = ?",
        [(conv_ids[i], emails[i]["id"]) for i in range(n)],
    )
    conn.commit()
    print(f"  {n:,} emailov aktualizovaných\n", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # REPORT
    # ─────────────────────────────────────────────────────────────────────────

    # Build groupings
    tid_to_eids: dict[str, list[int]] = defaultdict(list)
    cid_to_eids: dict[int, list[int]] = defaultdict(list)
    eid_to_cid:  dict[int, int]       = {}

    for i, em in enumerate(emails):
        eid = em["id"]
        tid = em["thread_id"] or f"__none_{eid}"
        cid = conv_ids[i]
        tid_to_eids[tid].append(eid)
        cid_to_eids[cid].append(eid)
        eid_to_cid[eid] = cid

    n_thread_groups = len(tid_to_eids)
    n_conv_groups   = len(cid_to_eids)

    # Size distribution helper
    def size_dist(groups: dict) -> dict:
        sizes = [len(v) for v in groups.values()]
        return {
            "1":    sum(1 for s in sizes if s == 1),
            "2-5":  sum(1 for s in sizes if 2 <= s <= 5),
            "6-20": sum(1 for s in sizes if 6 <= s <= 20),
            "20+":  sum(1 for s in sizes if s > 20),
            "max":  max(sizes),
        }

    t_dist = size_dist(tid_to_eids)
    c_dist = size_dist(cid_to_eids)

    print("=" * 70)
    print("  REPORT: thread_id vs conversation_id")
    print("=" * 70)
    print()
    print(f"  {'Metrika':<35} {'thread_id':>10}  {'conv_id':>10}")
    print("  " + "-" * 60)
    print(f"  {'Počet skupín':<35} {n_thread_groups:>10,}  {n_conv_groups:>10,}")
    print(f"  {'Skupiny s 1 mailom':<35} {t_dist['1']:>10,}  {c_dist['1']:>10,}")
    print(f"  {'Skupiny 2–5 mailov':<35} {t_dist['2-5']:>10,}  {c_dist['2-5']:>10,}")
    print(f"  {'Skupiny 6–20 mailov':<35} {t_dist['6-20']:>10,}  {c_dist['6-20']:>10,}")
    print(f"  {'Skupiny 20+ mailov':<35} {t_dist['20+']:>10,}  {c_dist['20+']:>10,}")
    print(f"  {'Najväčšia skupina':<35} {t_dist['max']:>10,}  {c_dist['max']:>10,}")
    print()

    # Emails where grouping differs (i.e. emails from the same thread_id
    # landed in different conv_ids, or vice versa)
    changed = 0
    for tid, eids in tid_to_eids.items():
        cids_in_tid = {eid_to_cid[e] for e in eids}
        if len(cids_in_tid) > 1:
            changed += len(eids)  # these emails got "split" by new method

    # Also count merges (different thread_ids now in same conv)
    merged_cross_tids = 0
    for cid, eids in cid_to_eids.items():
        tids_in_cid = {emails[idx_map[e]]["thread_id"] for e in eids
                       if emails[idx_map[e]]["thread_id"]}
        if len(tids_in_cid) > 1:
            merged_cross_tids += len(eids)

    print(f"  Emaily kde sa grouping líši:")
    print(f"    SPLIT (thread_id zlúčil, conv rozdelil):  {changed:,} emailov")
    print(f"    MERGE (thread_id rozdelil, conv zlúčil):  {merged_cross_tids:,} emailov")
    print()

    # ── MERGE examples (conv joined different thread_ids) ─────────────────────
    print("  --- MERGE príklady: conv_id zlúčilo čo thread_id rozdelil ---")
    merge_examples = []
    for cid, eids in cid_to_eids.items():
        tid_groups: dict[str, list] = defaultdict(list)
        for eid in eids:
            em  = emails[idx_map[eid]]
            tid = em["thread_id"] or ""
            if tid:
                tid_groups[tid].append(em)
        if len(tid_groups) < 2:
            continue
        merge_examples.append((cid, len(eids), tid_groups))

    merge_examples.sort(key=lambda x: -x[1])
    for cid, total, tid_groups in merge_examples[:3]:
        subj_sample = emails[idx_map[cid_to_eids[cid][0]]]["subject"] or ""
        print(f"\n  conv_id={cid}  ({total} emailov)  subject: '{subj_sample[:55]}'")
        for tid, tems in list(tid_groups.items())[:4]:
            dates_str = sorted(e["date"][:10] for e in tems if e["date"])
            d_range = f"{dates_str[0]}–{dates_str[-1]}" if dates_str else "?"
            print(f"    thread_id {tid[:40]}  "
                  f"({len(tems)} mailov, {d_range})")

    print()

    # ── SPLIT examples (thread_id was one group, conv split it) ───────────────
    print("  --- SPLIT príklady: conv_id rozdelilo čo thread_id zlúčil ---")
    split_examples = []
    for tid, eids in tid_to_eids.items():
        if tid.startswith("__none_"):
            continue
        cid_groups: dict[int, list] = defaultdict(list)
        for eid in eids:
            cid_groups[eid_to_cid[eid]].append(emails[idx_map[eid]])
        if len(cid_groups) < 2:
            continue
        split_examples.append((tid, len(eids), cid_groups))

    split_examples.sort(key=lambda x: -x[1])
    for tid, total, cid_groups in split_examples[:3]:
        tid_short = tid[:50]
        print(f"\n  thread_id {tid_short}  ({total} emailov)")
        for cid, cems in sorted(cid_groups.items())[:4]:
            subj_sample  = cems[0]["subject"] or ""
            dates_sorted = sorted(e["date"][:10] for e in cems if e["date"])
            d_range = f"{dates_sorted[0]}–{dates_sorted[-1]}" if dates_sorted else "?"
            first_p = next((e["from_address"] for e in cems if e["from_address"]), "?")
            print(f"    conv_id={cid}  ({len(cems)} mailov, {d_range})  "
                  f"subject: '{subj_sample[:40]}'  from: {first_p}")

    print()

    # ── size distribution detail ───────────────────────────────────────────────
    print("  --- Distribúcia veľkostí konverzácií (conversation_id) ---")
    all_sizes = sorted(len(v) for v in cid_to_eids.values())
    buckets = [(1, 1), (2, 5), (6, 20), (21, 50), (51, 9999)]
    for lo, hi in buckets:
        cnt   = sum(1 for s in all_sizes if lo <= s <= hi)
        mails = sum(s for s in all_sizes if lo <= s <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 9999 else f"{lo}+")
        print(f"    {label:>6} mailov/konv: {cnt:>6,} konverzácií  ({mails:>7,} mailov)")

    print(f"\n  Max konverzácia: {max(all_sizes)} mailov")
    conn.close()
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
