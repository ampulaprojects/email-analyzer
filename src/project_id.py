"""
Deterministic project identification from email conversation data.

NO LLM. Three signals in priority order:
  1. Known project name in subjects  (strongest — explicit label)
  2. Project code in subjects         (strong — internal code convention)
  3. Cluster label keywords           (fallback — inferred from semantic cluster)

Returns (project_name, source) where source is one of:
  "name" | "code" | "cluster" | "unknown"

Usage:
    from src.project_id import identify_project
    project, source = identify_project(subjects, cluster_label="Westend rezidencia")
"""

import re
from collections import Counter


# ── Signal 1: known project names ─────────────────────────────────────────────
# Order matters: longer / more specific patterns first to avoid partial matches.
# Each entry: (canonical_name, [regex_patterns])

_KNOWN_PROJECTS: list[tuple[str, list[str]]] = [
    ("One Eurovea",        [r"one.eurovea", r"one eurovea"]),
    ("Tower 220",          [r"T[-_]?220", r"Tower\s+220", r"\b2406\b.*tower|\btower\b.*\b2406\b"]),
    ("Westend",            [r"\bwestend\b", r"\bWR\b(?!\s*\d{4}-)"]),   # WR but not "WR 2024-..."
    ("Lido",               [r"\blido\b"]),
    ("Klingerka",          [r"\bklingerka\b", r"\bKLG\b"]),
    ("Patronka",           [r"\bpatronka\b"]),
    ("Pasienky",           [r"\bpasienky\b"]),
    ("Ihla",               [r"\bihla\b"]),
    ("Helios",             [r"\bhelios\b"]),
    ("Skypark",            [r"\bskypark\b"]),
    ("Pulsar",             [r"\bpulsar\b"]),
    ("Fuxton",             [r"\bfuxton\b", r"district\s*15"]),
    ("Vydrica",            [r"\bvydrica\b"]),
    ("Zimný prístav",      [r"zimn\w*\s+pr[ií]stav"]),
    ("Eurovea",            [r"\beurovea\b"]),          # generic Eurovea (no "One" prefix)
    ("Tower",              [r"\btower\b"]),             # generic Tower fallback
]

# Pre-compile all patterns
_NAME_RULES: list[tuple[str, re.Pattern]] = [
    (name, re.compile("|".join(pats), re.IGNORECASE))
    for name, pats in _KNOWN_PROJECTS
]


# ── Signal 2: project codes ────────────────────────────────────────────────────
# 4-digit GFI internal codes: YYSQ (year YY, sequence SQ)
# We capture the code and optionally the project name adjacent to it.

_CODE_RE = re.compile(r"\b(\d{4})[\W_]", re.IGNORECASE)

# Known code→project mappings (populated from observed data)
_CODE_MAP: dict[str, str] = {
    "2202": "Westend",
    "2501": "VH",           # Vojenská Historia area
    "2502": "NL",
    "2604": "One Eurovea",  # also used for Tower 220 subtasks — resolved by Signal 1
    "2406": "One Eurovea",
    "2607": "Lido",
}


# ── Signal 3: cluster label keywords ──────────────────────────────────────────

_CLUSTER_RULES: list[tuple[str, re.Pattern]] = [
    ("One Eurovea",  re.compile(r"one.eurovea|eurovea.*inf", re.IGNORECASE)),
    ("Tower 220",    re.compile(r"tower\s*220|T[-_]?220", re.IGNORECASE)),
    ("Westend",      re.compile(r"westend|WR_situacia", re.IGNORECASE)),
    ("Lido",         re.compile(r"\blido\b", re.IGNORECASE)),
    ("Klingerka",    re.compile(r"klingerka|KLG", re.IGNORECASE)),
    ("Patronka",     re.compile(r"patronka", re.IGNORECASE)),
    ("Pasienky",     re.compile(r"pasienky", re.IGNORECASE)),
    ("Ihla",         re.compile(r"\bihla\b", re.IGNORECASE)),
    ("Helios",       re.compile(r"\bhelios\b", re.IGNORECASE)),
    ("Skypark",      re.compile(r"skypark", re.IGNORECASE)),
    ("Vydrica",      re.compile(r"vydrica", re.IGNORECASE)),
]


# ── Public API ─────────────────────────────────────────────────────────────────

def identify_project(
    subjects: list[str],
    cluster_label: str | None = None,
) -> tuple[str, str]:
    """
    Identify project from a list of email subjects + optional cluster label.

    Returns (project_name, source):
      source = "name" | "code" | "cluster" | "unknown"

    Signal priority: name > code > cluster > unknown.
    Never guesses — returns "neznámy" if no signal matches.
    """
    # Deduplicate and normalise subjects
    unique_subjects = list({(s or "").strip() for s in subjects if s})

    # ── Signal 1: known project name in subjects ───────────────────────────────
    name_hits: Counter[str] = Counter()
    for subj in unique_subjects:
        for project, pat in _NAME_RULES:
            if pat.search(subj):
                name_hits[project] += 1
                break   # one project per subject (first match wins)

    if name_hits:
        return name_hits.most_common(1)[0][0], "name"

    # ── Signal 2: project code in subjects ────────────────────────────────────
    code_hits: Counter[str] = Counter()
    for subj in unique_subjects:
        for m in _CODE_RE.finditer(subj):
            code = m.group(1)
            if code in _CODE_MAP:
                code_hits[_CODE_MAP[code]] += 1

    if code_hits:
        return code_hits.most_common(1)[0][0], "code"

    # ── Signal 3: cluster label ────────────────────────────────────────────────
    if cluster_label:
        for project, pat in _CLUSTER_RULES:
            if pat.search(cluster_label):
                return project, "cluster"

    return "neznámy", "unknown"
