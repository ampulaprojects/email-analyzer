"""Auto-label clusters using llama3.1:8b via Ollama."""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

import requests
from dotenv import load_dotenv

try:
    from .db import get_connection
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.db import get_connection

load_dotenv()

DB_PATH     = os.environ.get("DB_PATH", "data/emails.db")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
GEN_MODEL   = "llama3.1:8b"

SAMPLE_SIZE = 10   # representative emails per cluster

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Ollama ────────────────────────────────────────────────────────────────────

def check_ollama() -> None:
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Ollama server nebezi na {OLLAMA_BASE}\nSpusti: ollama serve")
    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(GEN_MODEL in m for m in models):
        raise RuntimeError(
            f"Model {GEN_MODEL!r} nie je stiahnuty.\n"
            f"Spusti: ollama pull {GEN_MODEL}\n"
            f"Dostupne: {models}"
        )


def _generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False, "format": "json"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ── JSON extraction ───────────────────────────────────────────────────────────

_LABEL_KEYS = ("label", "nazov", "name", "kratky_nazov_skupiny", "kratky_nazov",
               "skupina", "title")
_DESC_KEYS  = ("description", "popis", "jednovetovy_popis", "opis", "desc")


def _parse_label_json(raw: str) -> dict | None:
    """Parse model response into {"label": ..., "description": ...}.

    Tries multiple strategies and normalises any key variants the model may use.
    """
    def _normalise(d: dict) -> dict | None:
        if not isinstance(d, dict):
            return None
        label = next((str(d[k]).strip() for k in _LABEL_KEYS if k in d), None)
        desc  = next((str(d[k]).strip() for k in _DESC_KEYS  if k in d), "")
        if label:
            return {"label": label, "description": desc}
        return None

    # 1. direct parse
    try:
        result = _normalise(json.loads(raw))
        if result:
            return result
    except json.JSONDecodeError:
        pass

    # 2. extract first complete {...} block (handles leading/trailing text)
    m = re.search(r'\{[^{}]*\}', raw, re.S)
    if m:
        try:
            result = _normalise(json.loads(m.group()))
            if result:
                return result
        except json.JSONDecodeError:
            pass

    # 3. greedy {...} spanning multiple levels
    m = re.search(r'\{.*\}', raw, re.S)
    if m:
        try:
            result = _normalise(json.loads(m.group()))
            if result:
                return result
        except json.JSONDecodeError:
            pass

    return None


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_unlabeled_clusters(conn, cluster_id: int | None) -> list:
    """Return clusters that have no description yet (auto-label = 'cluster_N')."""
    if cluster_id is not None:
        return conn.execute(
            "SELECT id, label, size FROM clusters WHERE id = ?",
            (cluster_id,),
        ).fetchall()
    return conn.execute(
        "SELECT id, label, size FROM clusters"
        " WHERE description IS NULL"
        " ORDER BY size DESC"
    ).fetchall()


def _get_sample_emails(conn, cluster_db_id: int, n: int = SAMPLE_SIZE) -> list:
    """Return top-n emails by confidence for a given cluster DB id."""
    return conn.execute(
        """
        SELECT e.subject, e.body_snippet, ec.confidence
        FROM email_clusters ec
        JOIN emails e ON e.id = ec.email_id
        WHERE ec.cluster_id = ?
        ORDER BY ec.confidence DESC
        LIMIT ?
        """,
        (cluster_db_id, n),
    ).fetchall()


def _save_label(conn, cluster_db_id: int, label: str, description: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE clusters SET label = ?, description = ?, updated_at = ? WHERE id = ?",
        (label, description, now, cluster_db_id),
    )
    conn.commit()


# ── prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(emails: list) -> str:
    lines = []
    for i, row in enumerate(emails, 1):
        subject = (row["subject"] or "").strip()
        snippet = (row["body_snippet"] or "").strip()[:120]
        lines.append(f"{i}. {subject}")
        if snippet and snippet != subject:
            lines.append(f"   {snippet}")

    email_block = "\n".join(lines)

    return (
        "Analyze these emails from one cluster and respond ONLY with valid JSON.\n"
        "Required JSON keys (exactly): label, description\n"
        "- label: short group name, max 5 words, in Slovak\n"
        "- description: one sentence about what this group is, in Slovak\n\n"
        "Emails:\n"
        f"{email_block}\n\n"
        'Output format (use exactly these key names): {"label": "...", "description": "..."}'
    )


# ── core labeling logic ───────────────────────────────────────────────────────

def label_cluster(conn, cluster_db_id: int, cluster_label: str, size: int) -> bool:
    emails = _get_sample_emails(conn, cluster_db_id)
    if not emails:
        log.warning("Cluster id=%d nema ziadne emaily — preskakujem", cluster_db_id)
        return False

    prompt = _build_prompt(emails)

    log.info("Generujem nazov pre cluster id=%d (%d emailov)...", cluster_db_id, size)
    t0 = time.time()
    parsed = None
    raw = ""
    for attempt in range(1, 4):  # up to 3 tries
        try:
            raw = _generate(prompt)
        except requests.exceptions.Timeout:
            log.error("Timeout (pokus %d) pre cluster id=%d", attempt, cluster_db_id)
            continue
        except Exception as exc:
            log.error("Chyba (pokus %d) pre cluster id=%d: %s", attempt, cluster_db_id, exc)
            continue
        parsed = _parse_label_json(raw)
        if parsed and "label" in parsed:
            break
        log.warning("JSON parse zlyhal (pokus %d) pre cluster id=%d — skusam znova",
                    attempt, cluster_db_id)
    elapsed = time.time() - t0

    if parsed and "label" in parsed:
        new_label = str(parsed["label"]).strip()[:80]
        description = str(parsed.get("description", "")).strip()[:200]
    else:
        # fallback: first non-empty line as label
        first_line = next((l.strip() for l in raw.splitlines() if l.strip()), raw[:80])
        new_label   = first_line[:80]
        description = ""
        log.warning("JSON parse zlyhal po 3 pokusoch pre cluster id=%d", cluster_db_id)

    _save_label(conn, cluster_db_id, new_label, description)
    print(f"  Cluster {cluster_db_id} ({size} emailov): {new_label}  [{elapsed:.1f}s]")
    if description:
        print(f"    {description}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label clusters via Ollama llama3.1:8b")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Len tento cluster (DB id). Bez arg = vsetky bez description.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max pocet klastrov na spracovanie")
    args = parser.parse_args()

    check_ollama()
    log.info("Ollama OK — model %s", GEN_MODEL)

    conn     = get_connection(DB_PATH)
    clusters = _get_unlabeled_clusters(conn, args.cluster)

    if args.limit:
        clusters = clusters[: args.limit]

    if not clusters:
        msg = (
            f"Cluster id={args.cluster} nenajdeny." if args.cluster
            else "Vsetky clustery uz maju description."
        )
        log.info(msg)
        conn.close()
        return

    log.info("Labeling %d klastrov...", len(clusters))
    print()

    ok = err = 0
    t_start = time.time()

    for row in clusters:
        success = label_cluster(conn, row["id"], row["label"], row["size"])
        if success:
            ok += 1
        else:
            err += 1

    conn.close()
    elapsed = time.time() - t_start

    print()
    print("--- Label summary ---------------------------------------------")
    print(f"  Spracovanych  : {ok + err:>5}")
    print(f"  OK            : {ok:>5}")
    print(f"  Chyby         : {err:>5}")
    print(f"  Celkovy cas   : {elapsed:>5.1f} s")
    print("---------------------------------------------------------------")


if __name__ == "__main__":
    main()
