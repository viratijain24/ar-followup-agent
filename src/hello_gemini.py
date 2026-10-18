"""Smoke test: prove the API key works for chat and embeddings.

Run from the project root:  python -m src.hello_gemini
"""
import os
import time

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()  # copies NAME=value lines from .env into os.environ
    if not os.environ.get("GEMINI_API_KEY") or "your-api-key" in os.environ["GEMINI_API_KEY"]:
        raise SystemExit("GEMINI_API_KEY missing: put your key in .env (see .env.example)")

    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    client = genai.Client()  # reads GEMINI_API_KEY from the environment

    # 1) One chat call.
    start = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        input="In one sentence, what is an overdue invoice?",
    )
    elapsed = time.perf_counter() - start
    print(f"[{model}] {interaction.output_text.strip()}")
    print(f"latency: {elapsed:.2f}s")
    print(f"usage: {getattr(interaction, 'usage', None)}")

    # 2) Which embedding models can this key see?
    embed_models = [m.name for m in client.models.list() if "embed" in (m.name or "")]
    print("\nembedding models available:", embed_models)

    # 3) One embedding call with the configured model.
    embed_model = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")
    result = client.models.embed_content(model=embed_model, contents="Invoice INV-1007 is overdue.")
    vector = result.embeddings[0].values
    print(f"[{embed_model}] vector length {len(vector)}, first 3 values {vector[:3]}")


if __name__ == "__main__":
    main()
