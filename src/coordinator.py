"""Coordinator: a plain-Python plan-and-execute loop over the agents.

extract (LLM) -> reconcile (pandas) -> retrieve (RAG) -> draft (LLM) -> verify (rules, then LLM critic)
On a failed check the draft is regenerated with the failure reasons (max 2 retries), then flagged.
Every step is recorded in a per-invoice trace (agent, attempt, latency, tokens, outcome).
Run from the project root:  python -m src.coordinator [--no-cache] [--no-verifier] [--data data]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from src import llm
from src.draft import DraftContext, draft_email, load_customer_records
from src.extract import extract_all
from src.rag import Index, load_chunks, retrieve
from src.reconcile import ACTIONABLE_STAGES, load_payments, normalize_ref, reconcile
from src.verify import check_values, critique

AS_OF = date(2026, 10, 18)  # the dataset's fixed "today"
MAX_RETRIES = 2
NO_ACTION_REASONS = {"settled": "paid in full", "not_due": "not yet due",
                     "grace": "in the 1-6 day grace period (policy section 1)"}


@dataclass
class Outcome:
    invoice_number: str
    status: str  # ready_to_send / needs_human_review / held / flagged / no_action
    reason: str = ""
    stage: str = ""
    draft: dict | None = None
    problems: list[str] = field(default_factory=list)
    first_draft: dict | None = None  # before any verifier feedback (for the with/without-verifier eval)
    retrieved: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


def _step(trace: list, agent: str, attempt: int, resp: llm.LLMResult | None, start: float, **info) -> None:
    trace.append({"agent": agent, "attempt": attempt, "latency_s": round(time.perf_counter() - start, 3),
                  "api_latency_s": resp.latency_s if resp else 0.0, "cached": resp.cached if resp else True,
                  "input_tokens": resp.input_tokens if resp else 0,
                  "output_tokens": (resp.output_tokens + resp.thought_tokens) if resp else 0, **info})


def process_invoice(ctx: DraftContext, use_verifier: bool = True, use_critic: bool = True) -> Outcome:
    row = ctx.row
    out = Outcome(row["invoice_number"], "flagged", stage=row["stage"],
                  retrieved=[c.chunk_id for c in ctx.retrieved.customer + ctx.retrieved.policy])
    feedback: list[str] = []
    for attempt in range(1 + MAX_RETRIES):
        t = time.perf_counter()
        draft, resp = draft_email(ctx, feedback)
        _step(out.trace, "draft", attempt, resp, t, action=draft.action)
        out.draft = draft.model_dump()
        out.first_draft = out.first_draft or out.draft
        if draft.action == "hold_for_review":
            out.status, out.reason = "held", draft.hold_reason
            return out
        if not use_verifier:
            break
        t = time.perf_counter()
        check = check_values(draft, ctx)
        _step(out.trace, "verify_rules", attempt, None, t, ok=check.ok, problems=check.problems)
        if not check.ok:
            feedback = out.problems = check.problems
            continue
        if use_critic:
            t = time.perf_counter()
            verdict, resp = critique(draft, ctx)
            _step(out.trace, "verify_critic", attempt, resp, t, ok=verdict.passes, problems=verdict.issues)
            if not verdict.passes:
                feedback = out.problems = verdict.issues
                continue
        out.problems = []
        break
    else:  # every attempt failed a check
        out.status, out.reason = "flagged", "failed verification after retries; needs a human"
        return out
    if row["stage"] == "final":
        out.status, out.reason = "needs_human_review", "final notice: policy section 5 requires human review"
    else:
        out.status, out.reason = "ready_to_send", ""
    return out


def prepare(data_dir: Path = Path("data"), as_of: date = AS_OF,
            verbose: bool = True) -> tuple[list[Outcome], list[tuple[DraftContext, dict]]]:
    """Extract, reconcile and retrieve. Returns (finished outcomes, (context, extract step) per actionable invoice)."""
    t = time.perf_counter()
    extractions = extract_all(data_dir, verbose=False)
    extract_trace = {r.invoice["invoice_number"] if r.invoice else r.pdf_file: r for r in extractions}
    invoices = [{**r.invoice, "customer_id": r.customer_id} for r in extractions if r.invoice and r.customer_id]
    outcomes = [Outcome(r.pdf_file, "flagged", reason=f"extraction failed: {r.errors or 'unknown customer'}")
                for r in extractions if not (r.invoice and r.customer_id)]
    if verbose:
        print(f"extracted {len(invoices)}/{len(extractions)} invoices in {time.perf_counter() - t:.1f}s", flush=True)

    payments = load_payments(data_dir / "payments.csv")
    table, unapplied = reconcile(invoices, payments, as_of)
    if len(unapplied) and verbose:
        print(f"warning: {len(unapplied)} payments could not be applied", flush=True)
    payments["invoice_number"] = payments["invoice_reference"].map(normalize_ref)
    customers = load_customer_records(data_dir / "customers.csv")
    by_number = {i["invoice_number"]: i for i in invoices}
    index = Index(load_chunks())

    contexts = []
    for row in table.to_dict("records"):
        inv_no = row["invoice_number"]
        ex = extract_trace[inv_no]
        ex_step = {"agent": "extract", "attempt": ex.attempts - 1, "api_latency_s": ex.latency_s, "cached": ex.cached,
                   "input_tokens": ex.input_tokens, "output_tokens": ex.output_tokens + ex.thought_tokens}
        if row["stage"] not in ACTIONABLE_STAGES:
            outcomes.append(Outcome(inv_no, "no_action", NO_ACTION_REASONS[row["stage"]], row["stage"], trace=[ex_step]))
            continue
        invoice = by_number[inv_no]
        mine = payments[(payments["invoice_number"] == inv_no) & (payments["currency"] == invoice["currency"])]
        contexts.append((DraftContext(row, invoice, customers[row["customer_id"]], mine.to_dict("records"),
                                      retrieve(index, row, invoice), as_of), ex_step))
    return outcomes, contexts


def run(data_dir: Path = Path("data"), as_of: date = AS_OF, use_verifier: bool = True,
        use_critic: bool = True, verbose: bool = True) -> list[Outcome]:
    outcomes, contexts = prepare(data_dir, as_of, verbose)
    for ctx, ex_step in contexts:
        inv_no, row = ctx.row["invoice_number"], ctx.row
        out = process_invoice(ctx, use_verifier, use_critic)
        out.trace.insert(0, ex_step)
        outcomes.append(out)
        if verbose:
            print(f"{inv_no} [{row['stage']}] -> {out.status}"
                  f"{' (' + out.reason + ')' if out.reason else ''}, {len([s for s in out.trace if s['agent'] == 'draft'])} draft(s)",
                  flush=True)
    return sorted(outcomes, key=lambda o: o.invoice_number)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=AS_OF)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-verifier", action="store_true", help="accept the first draft (baseline)")
    parser.add_argument("--out", type=Path, default=Path("outputs/run.json"))
    args = parser.parse_args()
    llm.set_cache(not args.no_cache, embeddings=True)  # the index embeddings are a one-time build cost
    outcomes = run(args.data, args.as_of, use_verifier=not args.no_verifier)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(o) for o in outcomes], indent=2, default=str), encoding="utf-8")
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    print(f"\n{counts}  -> {args.out}")


if __name__ == "__main__":
    main()
