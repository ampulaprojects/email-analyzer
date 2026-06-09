"""Quick smoke test for local Ollama API — run after installing Ollama and pulling models."""

import sys
import requests

OLLAMA_BASE = "http://localhost:11434"


def check_server() -> list[str]:
    """Return list of available model names, or raise if server unreachable."""
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def test_embedding(text: str = "test email subject") -> list[float]:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def test_generate(prompt: str = "Odpovedz jednym slovom: funguje?") -> str:
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def main() -> None:
    print("--- Ollama smoke test ---\n")

    print("1. Kontrola servera...")
    try:
        models = check_server()
    except requests.exceptions.ConnectionError:
        print("   CHYBA: Ollama server nebezi na localhost:11434")
        print("   Spusti: ollama serve  (alebo restartuj Ollama)")
        sys.exit(1)

    print(f"   OK — dostupne modely: {models or '(ziadne)'}")
    if not models:
        print("   Stiahni modely: ollama pull nomic-embed-text && ollama pull llama3.1:8b")
        sys.exit(1)

    print("\n2. Embedding (nomic-embed-text)...")
    if not any("nomic-embed-text" in m for m in models):
        print("   CHYBA: model nomic-embed-text nie je stiahnuty")
        print("   Spusti: ollama pull nomic-embed-text")
        sys.exit(1)
    vec = test_embedding("test email subject")
    print(f"   Dlzka vektora: {len(vec)}  (ocakavane: 768)")
    print(f"   Prve 3 hodnoty: {vec[:3]}")
    assert len(vec) == 768, f"Ocakavanych 768 dimenzii, dostal {len(vec)}"
    print("   OK")

    print("\n3. Generovanie textu (llama3.1:8b)...")
    if not any("llama3.1:8b" in m for m in models):
        print("   CHYBA: model llama3.1:8b nie je stiahnuty")
        print("   Spusti: ollama pull llama3.1:8b")
        sys.exit(1)
    answer = test_generate("Odpovedz jednym slovom: funguje?")
    print(f"   Odpoved: {answer!r}")
    print("   OK")

    print("\n--- Vsetky testy prebehli uspesne ---")


if __name__ == "__main__":
    main()
