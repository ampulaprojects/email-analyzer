"""Entity graph: person co-occurrence graph with automatic community detection."""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

try:
    from .db import get_connection
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.db import get_connection

DB_PATH         = os.environ.get("DB_PATH", "data/emails.db")
INTERNAL_DOMAIN = "gfi.sk"

# Edge weights
W_THREAD  = 3.0   # weight per shared thread (strong signal)
W_EMAIL   = 1.0   # weight per shared email  (weaker signal)
W_INT_INT = 0.2   # multiplier for internal-internal edges
           # (GFI people appear in everything → weaken their mutual links
           #  so communities form around external project contacts)

MIN_EDGE_WEIGHT    = 2.0
MIN_COMMUNITY_SIZE = 3

BLOCKLIST_RE = re.compile(
    r"no[-_]?reply|noreply|mailer.daemon|postmaster|bounce|"
    r"faktury|recepcia|deepl|notification|asana|smartsheet|"
    r"unsubscribe|donotreply|do.not.reply|auto.reply|autoreply",
    re.I,
)

STOP_WORDS = {
    # Slovak particles/prepositions
    "re", "fw", "fwd", "vs", "resp",
    "ak", "na", "sa", "je", "zo", "do", "od", "pre", "pri", "tak",
    "ale", "sme", "aj", "po", "za", "ten", "to", "ako", "nie", "vo",
    "by", "si", "mi", "mu", "ho", "ich", "nam", "vas",
    "som", "ste", "bol", "bola", "bolo", "boli", "bude",
    "len", "ani", "lebo", "alebo",
    # English
    "the", "of", "in", "and", "to", "for", "with", "from", "is",
    "are", "was", "be", "has", "have", "that", "this", "will",
    "at", "on", "or", "not", "but", "can", "its",
    # German
    "de", "und", "von",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_system(addr: str) -> bool:
    local = addr.split("@")[0] if "@" in addr else addr
    return bool(BLOCKLIST_RE.search(local))


def _domain(addr: str) -> str:
    return addr.split("@", 1)[1].lower() if "@" in addr else ""


def _is_internal(addr: str) -> bool:
    return _domain(addr) == INTERNAL_DOMAIN


def _parse_addrs(json_str) -> list[str]:
    if not json_str:
        return []
    try:
        return [
            a.get("address", "").lower().strip()
            for a in json.loads(json_str)
            if a.get("address", "").strip()
        ]
    except Exception:
        return []


def _email_persons(row) -> list[str]:
    """All valid, non-system persons in one email row."""
    addrs: set[str] = set()
    s = (row["from_address"] or "").lower().strip()
    if s and "@" in s and not _is_system(s):
        addrs.add(s)
    for a in _parse_addrs(row["to_addresses"]) + _parse_addrs(row["cc_addresses"]):
        if a and "@" in a and not _is_system(a):
            addrs.add(a)
    return sorted(addrs)


# ── graph building ────────────────────────────────────────────────────────────

def build_graph(db_path: str = DB_PATH) -> tuple[nx.Graph, dict]:
    """Build weighted co-occurrence graph from all emails."""
    conn = get_connection(db_path)
    print("Nacitavam emaily ...", flush=True)
    rows = conn.execute(
        "SELECT id, thread_id, from_address, to_addresses, cc_addresses, subject, date "
        "FROM emails ORDER BY date"
    ).fetchall()
    conn.close()
    print(f"  {len(rows)} emailov")

    # ── pass 1: parse persons per email ───────────────────────────────────────
    email_persons: dict[int, list[str]] = {}
    thread_persons: dict[str, set[str]] = defaultdict(set)
    person_emails:  dict[str, set[int]] = defaultdict(set)
    email_subject:  dict[int, str]      = {}
    email_date:     dict[int, str]      = {}

    for row in rows:
        eid     = row["id"]
        persons = _email_persons(row)
        email_persons[eid] = persons
        email_subject[eid] = row["subject"] or ""
        email_date[eid]    = (row["date"] or "")[:10]
        for p in persons:
            person_emails[p].add(eid)
        if row["thread_id"]:
            thread_persons[row["thread_id"]].update(persons)

    all_persons = set(person_emails)
    print(f"  Unikatnych osob : {len(all_persons)}")
    print(f"  Vlaknien        : {len(thread_persons)}")

    # ── pass 2: count pair co-occurrences ────────────────────────────────────
    thread_co: Counter = Counter()  # (a, b) → shared-thread count
    email_co:  Counter = Counter()  # (a, b) → shared-email count

    print("Pocitam thread co-occurrence ...", flush=True)
    for pset in thread_persons.values():
        plist = sorted(pset)
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                thread_co[(plist[i], plist[j])] += 1

    print("Pocitam email co-occurrence ...", flush=True)
    for persons in email_persons.values():
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                email_co[(persons[i], persons[j])] += 1

    # ── pass 3: build networkx graph ──────────────────────────────────────────
    print("Staviam graf ...", flush=True)
    G = nx.Graph()
    for p in all_persons:
        G.add_node(p, internal=_is_internal(p), domain=_domain(p))

    for pair in set(thread_co) | set(email_co):
        a, b = pair
        w = W_THREAD * thread_co[pair] + W_EMAIL * email_co[pair]
        if _is_internal(a) and _is_internal(b):
            w *= W_INT_INT
        if w >= MIN_EDGE_WEIGHT:
            G.add_edge(a, b, weight=w)

    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    print(
        f"  Uzlov: {G.number_of_nodes()}  |  "
        f"Hran: {G.number_of_edges()}  |  "
        f"Odstranenych izolovanych: {len(isolated)}"
    )

    meta = {
        "person_emails": person_emails,
        "email_subject": email_subject,
        "email_date":    email_date,
    }
    return G, meta


# ── community detection ───────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> list:
    """Louvain with networkx fallback to greedy modularity."""
    method = "?"
    try:
        from networkx.algorithms.community import louvain_communities
        coms   = louvain_communities(G, weight="weight", seed=42)
        method = "Louvain (networkx)"
    except Exception:
        try:
            import community as cm
            partition = cm.best_partition(G, weight="weight", random_state=42)
            groups: dict[int, set] = defaultdict(set)
            for node, cid in partition.items():
                groups[cid].add(node)
            coms   = list(groups.values())
            method = "Louvain (python-louvain)"
        except ImportError:
            from networkx.algorithms.community import greedy_modularity_communities
            coms   = list(greedy_modularity_communities(G, weight="weight"))
            method = "greedy modularity"

    print(f"  Metoda           : {method}")
    print(f"  Komunít celkom   : {len(coms)}")
    return coms


# ── hub pruning ──────────────────────────────────────────────────────────────

ALWAYS_EXCLUDE = frozenset({"tupek@gfi.sk"})


def prune_hubs(G: nx.Graph, n: int) -> nx.Graph:
    """Remove top-n nodes by degree centrality + ALWAYS_EXCLUDE set.

    Returns a copy of G with those nodes removed.
    """
    centrality = nx.degree_centrality(G)
    top_n = {p for p, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:n]}
    to_remove = (top_n | ALWAYS_EXCLUDE) & set(G.nodes())

    print(f"\n  Vylucene huby ({len(to_remove)}):")
    for p in sorted(to_remove, key=lambda x: centrality.get(x, 0), reverse=True):
        print(
            f"    {p:<42}  centralita={centrality.get(p, 0):.4f}"
            f"  weighted_deg={G.degree(p, weight='weight'):.0f}"
        )

    H = G.copy()
    H.remove_nodes_from(to_remove)
    print(
        f"  Graf po vyluceni: {H.number_of_nodes()} uzlov  |  "
        f"{H.number_of_edges()} hran"
    )
    return H


# ── community analysis ────────────────────────────────────────────────────────

def _top_words(subjects: list[str], n: int = 12) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for s in subjects:
        for w in re.findall(
            r'\b[a-zA-ZáäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ]{3,}\b', s
        ):
            w = w.lower()
            if w not in STOP_WORDS:
                counter[w] += 1
    return counter.most_common(n)


def analyze_communities(G: nx.Graph, raw_coms: list, meta: dict) -> list[dict]:
    person_emails = meta["person_emails"]
    email_subject = meta["email_subject"]
    email_date    = meta["email_date"]
    centrality    = nx.degree_centrality(G)

    results: list[dict] = []
    for comm in raw_coms:
        if len(comm) < MIN_COMMUNITY_SIZE:
            continue

        internal = [p for p in comm if     _is_internal(p)]
        external = [p for p in comm if not _is_internal(p)]
        top5     = sorted(comm, key=lambda p: centrality.get(p, 0), reverse=True)[:5]
        ext_doms = Counter(_domain(p) for p in external if _domain(p)).most_common(5)

        # all emails where ANY member appears
        all_eids: set[int] = set()
        for p in comm:
            all_eids.update(person_emails.get(p, set()))

        subjects  = [email_subject[e] for e in all_eids if e in email_subject]
        top_words = _top_words(subjects)

        dates = sorted(
            email_date[e] for e in all_eids if e in email_date and email_date[e]
        )
        results.append(dict(
            size      = len(comm),
            n_int     = len(internal),
            n_ext     = len(external),
            top5      = top5,
            ext_doms  = ext_doms,
            top_words = top_words,
            first     = dates[0][:7] if dates else "",
            last      = dates[-1][:7] if dates else "",
            n_emails  = len(all_eids),
            members   = comm,
        ))

    results.sort(key=lambda x: x["size"], reverse=True)
    return results


# ── display ───────────────────────────────────────────────────────────────────

def print_communities(results: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(f"  KOMUNITY  —  {len(results)} komunit (min {MIN_COMMUNITY_SIZE} osob)")
    print(f"{'=' * 72}")

    for i, c in enumerate(results, 1):
        print(
            f"\n--- {i:>2}. komunita  "
            f"({c['size']} osob  {c['n_int']} INT / {c['n_ext']} EXT  |  "
            f"{c['n_emails']} emailov  |  {c['first']} – {c['last']}) ---"
        )
        print("  Top 5 (centralita):")
        for p in c["top5"]:
            tag = "INT" if _is_internal(p) else "EXT"
            print(f"    [{tag}]  {p}")
        if c["ext_doms"]:
            doms = "  ".join(f"{d}({n})" for d, n in c["ext_doms"])
            print(f"  Ext. domeny : {doms}")
        if c["top_words"]:
            words = "  ".join(f"{w}({n})" for w, n in c["top_words"])
            print(f"  Temy        : {words}")


def print_members(results: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print("  CLENOVIA KOMUNÍT")
    print(f"{'=' * 72}")
    for i, c in enumerate(results, 1):
        print(f"\n--- {i:>2}. komunita ---")
        for p in sorted(c["members"]):
            tag = "INT" if _is_internal(p) else "EXT"
            print(f"  [{tag}]  {p}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global MIN_EDGE_WEIGHT, MIN_COMMUNITY_SIZE

    parser = argparse.ArgumentParser(description="Person co-occurrence graph + community detection")
    parser.add_argument("--db",            default=DB_PATH,          help="SQLite DB path")
    parser.add_argument("--min-weight",    type=float, default=MIN_EDGE_WEIGHT,
                        help=f"Min edge weight (default {MIN_EDGE_WEIGHT})")
    parser.add_argument("--min-community", type=int,   default=MIN_COMMUNITY_SIZE,
                        help=f"Min community size (default {MIN_COMMUNITY_SIZE})")
    parser.add_argument("--show-members",  action="store_true",
                        help="Print all members of each community")
    parser.add_argument("--exclude-hubs",  type=int, default=0,
                        help="Exclude top N hub nodes before community detection (0 = off)")
    args = parser.parse_args()

    MIN_EDGE_WEIGHT    = args.min_weight
    MIN_COMMUNITY_SIZE = args.min_community

    print("=== Graf entit ===")
    G, meta = build_graph(args.db)

    if args.exclude_hubs > 0:
        print(f"\n=== Pruning top-{args.exclude_hubs} hubov + always {list(ALWAYS_EXCLUDE)} ===")
        G = prune_hubs(G, args.exclude_hubs)

    print("\n=== Detekcia komunít ===")
    raw  = detect_communities(G)
    coms = analyze_communities(G, raw, meta)
    print(f"  Po filtraci (>= {MIN_COMMUNITY_SIZE}): {len(coms)} komunít")

    print_communities(coms)

    if args.show_members:
        print_members(coms)


if __name__ == "__main__":
    main()
