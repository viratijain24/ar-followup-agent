"""RAG over the collections corpus: chunk, embed (gemini-embedding-2), index in FAISS, retrieve.

Metadata where the rule depends on a known fact, search where it's buried in text.
1. Customer context: the customer's notes (by customer_id), plus the top-k of that customer's emails.
   Emails are ranked by HYBRID search: BM25 keyword ranking + vector ranking, merged with Reciprocal
   Rank Fusion. Emails are treated as untagged, like real mail: the `invoices` / `reference` fields in
   their front matter are the answer key for eval/retrieval.py and are never read here.
2. Policy sections:
   - by metadata: sections tagged with the invoice's stage, sections tagged `always` (required
     content), and the partial-payments section when the invoice is partly paid;
   - by similarity: top-k of the remaining general (`all`) sections, using a query built from the
     invoice situation AND the customer context (a dispute in the emails pulls in the disputes rule).
   Sections tagged `reference` (the stage table) are enforced in code by reconcile.py, not retrieved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src import llm

CORPUS_DIR = Path("data/corpus")
RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al., 2009); not tuned
EMAIL_METHODS = ("keyword", "vector", "hybrid")


@dataclass
class Chunk:
    chunk_id: str
    doc_type: str  # policy / customer_note / correspondence
    title: str
    text: str
    customer_id: str | None = None
    stage: str | None = None  # policy only: friendly / firm / final / all / always / reference
    invoices: list[str] = field(default_factory=list)  # emails: answer key only
    reference: str | None = None  # emails: answer key only (direct / indirect / none)


def _front_matter(text: str) -> tuple[dict, str]:
    """Parse a minimal '---' YAML header of `key: value` and `key: [a, b]` lines."""
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        meta[key.strip()] = [v.strip() for v in value[1:-1].split(",") if v.strip()] if value.startswith("[") else value
    return meta, text[m.end():]


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks = []
    # Policy: one chunk per "## " section, tagged with its <!-- stage: ... --> comment.
    _, body = _front_matter((corpus_dir / "collections_policy.md").read_text(encoding="utf-8"))
    for section in re.split(r"\n(?=## )", body)[1:]:
        title = section.splitlines()[0].lstrip("# ").strip()
        stage = re.search(r"<!-- stage: (\w+) -->", section).group(1)
        text = re.sub(r"<!--.*?-->\n?", "", section).strip()
        chunks.append(Chunk(f"policy:{title.split('.')[0]}", "policy", title, text, stage=stage))
    # Customer notes and correspondence: one chunk per file (they're short).
    for path in sorted((corpus_dir / "customer_notes").glob("*.md")) + sorted((corpus_dir / "correspondence").glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        chunks.append(Chunk(f"{meta['doc_type']}:{path.stem}", meta["doc_type"], path.stem, body.strip(),
                            customer_id=meta.get("customer_id"), invoices=meta.get("invoices", []),
                            reference=meta.get("reference")))
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase words; hyphenated IDs stay whole ('INV-1008' -> 'inv-1008')."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class Index:
    """FAISS inner-product index over L2-normalized vectors (= cosine similarity)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # gemini-embedding-2 takes task prefixes instead of a task_type parameter.
        vectors = llm.embed([f"title: {c.title} | text: {c.text}" for c in chunks])
        self.matrix = self._normalize(np.array(vectors, dtype="float32"))
        self.faiss = faiss.IndexFlatIP(self.matrix.shape[1])
        self.faiss.add(self.matrix)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, query: str, allowed: set[int], k: int) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity, restricted to the `allowed` chunk positions."""
        q = self._normalize(np.array(llm.embed([f"task: search result | query: {query}"]), dtype="float32"))
        scores, ids = self.faiss.search(q, len(self.chunks))  # tiny corpus: score all, then filter
        hits = [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i in allowed]
        return hits[:k]

    def keyword_search(self, query: str, allowed: set[int]) -> list[tuple[Chunk, float]]:
        """All `allowed` chunks ranked by BM25, with the index built over just those chunks."""
        ids = sorted(allowed)
        bm25 = BM25Okapi([tokenize(self.chunks[i].text) for i in ids])
        scores = bm25.get_scores(tokenize(query))
        order = sorted(range(len(ids)), key=lambda j: -scores[j])  # stable: ties keep file (date) order
        return [(self.chunks[ids[j]], float(scores[j])) for j in order]


def rrf(*rankings: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: score(doc) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])  # Python's sort is stable: ties keep first-seen order


def email_query(row: dict, invoice: dict | None) -> str:
    """Describe the invoice the way an email might: number, month, items, total."""
    parts = [f"Invoice {row['invoice_number']}"]
    if invoice:
        issued = date.fromisoformat(str(invoice["issue_date"]))
        items = "; ".join(f"{li['quantity']:g} x {li['description']}" for li in invoice["line_items"])
        parts.append(f"issued {issued:%B %Y}, total {invoice['currency']} {invoice['total']:,.2f}, for {items}")
    return ", ".join(parts) + ". Payment, delay, dispute or promise to pay."


def rank_emails(index: Index, row: dict, invoice: dict | None, method: str = "hybrid") -> list[str]:
    """All of the customer's email chunk_ids, best first, by keyword / vector / hybrid (RRF)."""
    allowed = {i for i, c in enumerate(index.chunks)
               if c.doc_type == "correspondence" and c.customer_id == row["customer_id"]}
    if not allowed:
        return []
    query = email_query(row, invoice)
    keyword = [c.chunk_id for c, _ in index.keyword_search(query, allowed)]
    vector = [c.chunk_id for c, _ in index.search(query, allowed, len(allowed))]
    return {"keyword": keyword, "vector": vector, "hybrid": rrf(keyword, vector)}[method]


@dataclass
class Retrieved:
    customer: list[Chunk]
    policy: list[Chunk]
    query: str
    scores: dict[str, float]  # similarity scores of the general policy sections chosen
    email_ranking: list[str] = field(default_factory=list)  # all of the customer's emails, best first (for eval)


PARTIAL_SECTION = "policy:8"


def retrieve(index: Index, row: dict, invoice: dict | None = None, k_general: int = 2, k_email: int = 2,
             email_method: str = "hybrid") -> Retrieved:
    """`row` is one reconciled invoice (invoice_number, customer_id, status, stage, days_overdue, ...);
    `invoice` is its extracted data (issue_date, currency, total, line_items), used to find indirect emails."""
    by_id = {c.chunk_id: c for c in index.chunks}
    notes = [c for c in index.chunks if c.doc_type == "customer_note" and c.customer_id == row["customer_id"]]
    email_ranking = rank_emails(index, row, invoice, email_method)
    customer = notes + [by_id[cid] for cid in email_ranking[:k_email]]
    situation = (f"Invoice {row['invoice_number']} is {row['days_overdue']} days overdue, "
                 f"stage {row['stage']}, payment status {row['status']}.")
    query = situation + " Customer context: " + " ".join(c.text for c in customer)

    fixed = [c for c in index.chunks if c.doc_type == "policy" and
             (c.stage in (row["stage"], "always") or (c.chunk_id == PARTIAL_SECTION and row["status"] == "partial"))]
    general = {i for i, c in enumerate(index.chunks)
               if c.doc_type == "policy" and c.stage == "all" and c not in fixed}
    hits = index.search(query, general, k_general)
    return Retrieved(customer, fixed + [c for c, _ in hits], query, {c.chunk_id: s for c, s in hits}, email_ranking)


def main() -> None:
    import json

    from src.reconcile import ACTIONABLE_STAGES, load_payments, reconcile

    truth = json.loads(Path("data/ground_truth.json").read_text(encoding="utf-8"))
    invoices = {i["invoice_number"]: i for i in truth["invoices"]}
    table, _ = reconcile(truth["invoices"], load_payments(), date.fromisoformat(truth["as_of"]))
    index = Index(load_chunks())
    print(f"indexed {len(index.chunks)} chunks\n")
    for row in table[table["stage"].isin(ACTIONABLE_STAGES)].to_dict("records"):
        r = retrieve(index, row, invoices[row["invoice_number"]])
        print(f"{row['invoice_number']} ({row['customer_id']}, {row['stage']}, {row['status']})")
        print("  customer:", [c.chunk_id for c in r.customer])
        print("  policy:  ", [c.chunk_id for c in r.policy],
              "| general scores:", {k: round(v, 3) for k, v in r.scores.items()})


if __name__ == "__main__":
    main()
