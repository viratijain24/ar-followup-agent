"""Retrieval check on the dev set (seed 42) for every actionable invoice.

- Emails: retrieval never reads the `invoices` / `reference` fields in an email's front matter; here
  they are the answer key. recall@k = share of relevant emails ranked in the top k, compared for
  keyword-only (BM25), vector-only and hybrid (RRF), split by direct vs indirect reference.
- Policy (pipeline = hybrid emails): required sections must be present (section 7 always, section 8 if
  partially paid, section 9 for the disputed INV-1009, section 10 for INV-1005 with a payment promise).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.rag import EMAIL_METHODS, Index, load_chunks, rank_emails, retrieve
from src.reconcile import ACTIONABLE_STAGES, load_payments, reconcile

POLICY_MUST_HAVE = {"INV-1009": {"policy:9"}, "INV-1005": {"policy:10"}}
RELEVANT_GENERAL = {"policy:9": {"INV-1009"}, "policy:10": {"INV-1005"}}
KS = (1, 2)
TYPES = ("all", "direct", "indirect")


def _short(chunk_id: str) -> str:
    return chunk_id.split(":", 1)[1][11:]  # drop "correspondence:" and the date prefix


def run(data_dir: Path = Path("data")) -> dict:
    truth = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    invoices = {i["invoice_number"]: i for i in truth["invoices"]}
    table, _ = reconcile(truth["invoices"], load_payments(data_dir / "payments.csv"), date.fromisoformat(truth["as_of"]))
    index = Index(load_chunks())
    emails = {c.chunk_id: c for c in index.chunks if c.doc_type == "correspondence"}
    rows = table[table["stage"].isin(ACTIONABLE_STAGES)].to_dict("records")

    hits = {(m, k, t): 0 for m in EMAIL_METHODS for k in KS for t in TYPES}
    totals = {t: 0 for t in TYPES}
    ceiling = {k: 0 for k in KS}  # best achievable: an invoice with 3 relevant emails can't get all 3 in a top 2
    misses = {m: [] for m in EMAIL_METHODS}
    for row in rows:
        inv = row["invoice_number"]
        relevant = {cid: c.reference for cid, c in emails.items() if inv in c.invoices}
        for t in TYPES:
            totals[t] += sum(t in ("all", ref) for ref in relevant.values())
        for k in KS:
            ceiling[k] += min(k, len(relevant))
        for m in EMAIL_METHODS:
            ranking = rank_emails(index, row, invoices[inv], m)
            for cid, ref in relevant.items():
                for k in KS:
                    if cid in ranking[:k]:
                        hits[(m, k, "all")] += 1
                        hits[(m, k, ref)] += 1
                if cid not in ranking[:2]:
                    misses[m].append(f"{inv}: {ref} email '{_short(cid)}' ranked {ranking.index(cid) + 1}/"
                                     f"{len(ranking)}; top 2 = {[_short(c) for c in ranking[:2]]}")

    # Policy sections, as the pipeline retrieves them (hybrid emails feed the policy query).
    policy_found = policy_total = sem_relevant = sem_total = 0
    policy_misses = []
    for row in rows:
        inv = row["invoice_number"]
        r = retrieve(index, row, invoices[inv])
        need = POLICY_MUST_HAVE.get(inv, set()) | {"policy:7"} | ({"policy:8"} if row["status"] == "partial" else set())
        got = {c.chunk_id for c in r.policy}
        policy_found += len(need & got)
        policy_total += len(need)
        policy_misses += [f"{inv}: policy {c} missing" for c in sorted(need - got)]
        sem_total += len(r.scores)
        sem_relevant += sum(inv in RELEVANT_GENERAL.get(cid, set()) for cid in r.scores)

    print(f"\n=== Email retrieval, {len(rows)} actionable invoices, {totals['all']} relevant emails "
          f"({totals['direct']} direct, {totals['indirect']} indirect) ===")
    print(f"{'method':<10}" + "".join(f"{f'R@{k} {t}':>15}" for k in KS for t in TYPES))
    for m in EMAIL_METHODS:
        print(f"{m:<10}" + "".join(f"{f'{hits[(m, k, t)]}/{totals[t]}':>15}" for k in KS for t in TYPES))
    print(f"{'max':<10}" + "".join(f"{f'{ceiling[k]}/{totals[t]}' if t == 'all' else '':>15}" for k in KS for t in TYPES))
    print("(R@k = relevant emails ranked in the top k. 'max': invoices with 2-3 relevant emails cap R@1.)")
    for m in EMAIL_METHODS:
        for miss in misses[m]:
            print(f"  MISS@2 [{m}] {miss}")
    print(f"\nrequired policy sections retrieved (pipeline, hybrid): {policy_found}/{policy_total}")
    print(f"similarity-chosen policy sections that were relevant: {sem_relevant}/{sem_total}")
    for miss in policy_misses:
        print(f"  MISS {miss}")

    return {"invoices": len(rows), "relevant": totals, "ceiling": ceiling,
            "recall": {f"{m}@{k}_{t}": hits[(m, k, t)] / totals[t] for m in EMAIL_METHODS for k in KS for t in TYPES},
            "policy_required_recall": policy_found / policy_total,
            "policy_semantic_precision": sem_relevant / sem_total if sem_total else 0.0,
            "misses": misses, "policy_misses": policy_misses}
