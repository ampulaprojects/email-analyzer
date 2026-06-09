"""Generate embeddings for emails using Ollama nomic-embed-text."""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM   = 768
BATCH_SIZE  = 20       # DB commit interval
DEFAULT_WORKERS = 8    # parallel Ollama requests

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def _err_log() -> logging.Logger:
    err = logging.getLogger("embed.errors")
    if not err.handlers:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            Path(DB_PATH).parent / "errors_embed.log", encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        err.addHandler(fh)
        err.propagate = False
    return err


# ── Ollama helpers ────────────────────────────────────────────────────────────

def check_ollama() -> None:
    """Raise RuntimeError if Ollama is unreachable or model is missing."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Ollama server nebezi na {OLLAMA_BASE}\n"
            "Spusti: ollama serve"
        )
    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(EMBED_MODEL in m for m in models):
        raise RuntimeError(
            f"Model {EMBED_MODEL!r} nie je stiahnuty.\n"
            f"Spusti: ollama pull {EMBED_MODEL}\n"
            f"Dostupne modely: {models}"
        )


def embed_text(text: str) -> np.ndarray:
    """Return a float32 numpy array of shape (768,) for the given text."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    vector = resp.json()["embedding"]
    arr = np.array(vector, dtype=np.float32)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(f"Ocakavanych {EMBED_DIM} dimenzii, dostal {arr.shape}")
    return arr


# ── text builder ──────────────────────────────────────────────────────────────

def _build_input(row) -> str:
    parts = [
        (row["subject"]   or "").strip(),
        (row["from_name"] or "").strip(),
        (row["body_text"] or "")[:500].strip(),
    ]
    return " ".join(p for p in parts if p)


# ── worker (runs in thread) ───────────────────────────────────────────────────

def _embed_row(row) -> tuple[int, bytes | None, str | None]:
    """Returns (email_id, blob, error_message). blob is None on failure."""
    try:
        arr = embed_text(_build_input(row))
        return (row["id"], arr.tobytes(), None)
    except Exception as exc:
        return (row["id"], None, str(exc))


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_pending(conn, limit: int) -> list:
    return conn.execute(
        """
        SELECT id, subject, from_name, body_text
        FROM emails
        WHERE embedding IS NULL
        ORDER BY date ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _save_batch(conn, updates: list[tuple]) -> None:
    """updates = [(embedding_blob, email_id), ...]"""
    conn.executemany(
        "UPDATE emails SET embedding = ? WHERE id = ?",
        updates,
    )
    conn.commit()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings via Ollama")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Max emails to embed (default: all)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel Ollama requests (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    check_ollama()
    log.info("Ollama OK — model %s, workers=%d", EMBED_MODEL, args.workers)

    conn  = get_connection(DB_PATH)
    limit = args.limit or 999_999
    rows  = _get_pending(conn, limit)
    total = len(rows)

    if total == 0:
        log.info("Vsetky emaily uz maju embedding.")
        conn.close()
        return

    log.info("Generujem embeddingy pre %d emailov", total)

    err        = _err_log()
    ok_count   = 0
    err_count  = 0
    start_time = time.time()
    eta_logged = False
    updates: list[tuple] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # submit in sliding batches: BATCH_SIZE futures at a time
        for batch_start in range(0, total, BATCH_SIZE):
            batch = rows[batch_start : batch_start + BATCH_SIZE]
            futures = {pool.submit(_embed_row, row): row for row in batch}

            for fut in as_completed(futures):
                email_id, blob, error = fut.result()
                if blob is not None:
                    updates.append((blob, email_id))
                    ok_count += 1
                else:
                    row = futures[fut]
                    err.error("id=%d subject=%r: %s",
                              email_id, (row["subject"] or "")[:50], error)
                    err_count += 1

            # commit after each batch
            if updates:
                _save_batch(conn, updates)
                updates.clear()

            processed = min(batch_start + BATCH_SIZE, total)
            elapsed   = time.time() - start_time

            # ETA after first BATCH_SIZE * 5 emails
            if not eta_logged and processed >= min(100, total):
                avg     = elapsed / processed
                eta_sec = avg * (total - processed)
                log.info("ETA: ~%.0f min (priemer %.3f s/email, workers=%d)",
                         eta_sec / 60, avg, args.workers)
                eta_logged = True

            log.info("[%d/%d] embedding...  ok=%d err=%d  %.1fs",
                     processed, total, ok_count, err_count, elapsed)

    conn.close()

    total_time = time.time() - start_time
    avg_per    = total_time / total if total else 0

    print("\n--- Embedding summary -----------------------------------------")
    print(f"  Workers       : {args.workers:>6}")
    print(f"  Spracovanych  : {total:>6}")
    print(f"  OK            : {ok_count:>6}")
    print(f"  Chyby         : {err_count:>6}")
    print(f"  Celkovy cas   : {total_time:>6.1f} s  ({total_time/60:.1f} min)")
    print(f"  Priemer/email : {avg_per:>6.3f} s")
    print("---------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
