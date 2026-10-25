"""End-to-end pipeline eval on the dev set: hold decisions, verifier effectiveness, latency and cost.

Verifier effectiveness, two ways (CLAUDE.md):
- Natural: drafts with a wrong value, with vs. without the verifier, at the deployed setting
  (temperature 0.2) and in a stress sample (5 drafts per invoice at temperature 1.0, fixed seeds).
- Seeded: corrupt accepted drafts on purpose and report the share the verifier catches, plus its
  false-flag rate on clean drafts.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from src.coordinator import process_invoice, prepare
from src.draft import Draft, draft_email, fmt_date
from src.verify import check_values, critique

# Expected decisions, from the corpus (dispute on INV-1009; promise for INV-1005 dated after the as-of date).
EXPECTED_HOLD = {"INV-1005", "INV-1009"}
STRESS_SEEDS = (1, 2, 3, 4, 5)
STRESS_TEMPERATURE = 1.0
# gemini-3.5-flash-lite paid-tier prices per 1M tokens (ai.google.dev/gemini-api/docs/pricing, checked 2026-10-22).
PRICE_IN, PRICE_OUT = 0.30, 2.50
THREAT = "If payment is not received, we will refer this matter to our solicitors and a debt collection agency."
LATE_FEE = "Under our payment terms, a late-payment charge of 1.5% per month may apply to overdue balances."


def _with_body(draft: Draft, body: str) -> Draft:
    return draft.model_copy(update={"body": body})


def _swap_digits(s: str) -> str:
    """Swap the first pair of adjacent, different digits: '3,758.40' -> '3,578.40'."""
    for i in range(len(s) - 1):
        if s[i].isdigit() and s[i + 1].isdigit() and s[i] != s[i + 1] and "." not in s[:i + 1]:
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


def corruptions(draft: Draft, ctx, other_invoice: str) -> dict[str, Draft]:
    """One corrupted copy of `draft` per error type (types that can't apply to this draft are skipped)."""
    body, row, inv = draft.body, ctx.row, ctx.invoice
    balance, total, cur = f"{row['balance']:,.2f}", f"{inv['total']:,.2f}", inv["currency"]
    due = fmt_date(inv["due_date"])
    out = {
        "amount: swapped digits": body.replace(balance, _swap_digits(balance)),
        "amount: off by 100.00": body.replace(balance, f"{row['balance'] + 100:,.2f}"),
        "date: due date +3 days": body.replace(due, fmt_date(date.fromisoformat(inv["due_date"]) + timedelta(days=3))),
        "invoice no.: another of theirs": (body[:body.rfind(row["invoice_number"])] + other_invoice
                                           + body[body.rfind(row["invoice_number"]) + len(row["invoice_number"]):]),
        "currency: wrong code": body.replace(f"{cur} ", "EUR " if cur != "EUR" else "USD ", 1),
    }
    if row["status"] == "partial":
        out["role swap: total as balance"] = body.replace(balance, total)
    return {k: _with_body(draft, v) for k, v in out.items() if v != body}


def run(data_dir: Path = Path("data"), verbose: bool = True) -> dict:
    truth = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    by_customer = defaultdict(list)
    for t in truth["invoices"]:
        by_customer[t["customer_id"]].append(t["invoice_number"])
    finished, contexts = prepare(data_dir, verbose=False)

    # ---- 1. Deployed pipeline (temperature 0.2, verifier on) ----
    outcomes = {}
    for ctx, extract_step in contexts:
        out = process_invoice(ctx)
        out.trace.insert(0, extract_step)  # count extraction time too, as the coordinator does
        outcomes[ctx.row["invoice_number"]] = (ctx, out)
    held = {n for n, (_, o) in outcomes.items() if o.status == "held"}
    hold_correct = sum((n in held) == (n in EXPECTED_HOLD) for n in outcomes)
    statuses = Counter(o.status for _, o in outcomes.values()) + Counter(o.status for o in finished)
    sent = {n: (ctx, o) for n, (ctx, o) in outcomes.items() if o.status != "held"}
    first_flagged = [n for n, (ctx, o) in sent.items()
                     if not check_values(Draft(**o.first_draft), ctx).ok]
    final_flagged = [n for n, (ctx, o) in sent.items() if o.status == "flagged"]
    retries = sum(sum(s["agent"] == "draft" for s in o.trace) - 1 for _, o in outcomes.values())

    # ---- 2. Stress sample: 5 drafts per invoice at temperature 1.0, no verifier ----
    stress = []
    for n, (ctx, _) in outcomes.items():
        for seed in STRESS_SEEDS:
            d, _ = draft_email(ctx, temperature=STRESS_TEMPERATURE, seed=seed)
            check = check_values(d, ctx)
            crit = critique(d, ctx)[0] if d.action == "send" else None
            stress.append({"invoice": n, "seed": seed, "action": d.action, "rules_ok": check.ok,
                           "rules_problems": check.problems, "critic_ok": crit.passes if crit else None,
                           "critic_issues": crit.issues if crit else [], "draft": d.model_dump()})
    stress_holds_correct = sum((s["action"] == "hold_for_review") == (s["invoice"] in EXPECTED_HOLD) for s in stress)
    stress_sent = [s for s in stress if s["action"] == "send"]

    # ---- 3. Seeded corruptions of the accepted drafts ----
    seeded = defaultdict(lambda: [0, 0])  # type -> [caught, total]
    seeded_missed = []
    for n, (ctx, o) in sent.items():
        other = next(i for i in by_customer[ctx.row["customer_id"]] if i != n)
        for kind, bad in corruptions(Draft(**o.draft), ctx, other).items():
            caught = not check_values(bad, ctx).ok
            seeded[kind][0] += caught
            seeded[kind][1] += 1
            if not caught:
                seeded_missed.append(f"{n}: {kind}")
    clean_false_flags = sum(not check_values(Draft(**o.draft), ctx).ok for ctx, o in sent.values())
    tone = defaultdict(lambda: [0, 0])
    for n, (ctx, o) in sent.items():
        d = Draft(**o.draft)
        cases = {"legal threat added": THREAT}
        if ctx.row["stage"] == "friendly":
            cases["late-fee wording at friendly stage"] = LATE_FEE
        for kind, sentence in cases.items():
            bad = _with_body(d, d.body.replace("Accounts Receivable, Kestrel", f"{sentence}\n\nAccounts Receivable, Kestrel", 1))
            tone[kind][0] += not critique(bad, ctx)[0].passes
            tone[kind][1] += 1
    critic_clean_flags = sum(not critique(Draft(**o.draft), ctx)[0].passes for ctx, o in sent.values())

    # ---- 4. Latency and cost per invoice (all 20 invoices; API time only) ----
    traces = [o.trace for _, o in outcomes.values()] + [o.trace for o in finished]
    api_s = [sum(s["api_latency_s"] for s in tr) for tr in traces]
    tok_in = [sum(s["input_tokens"] for s in tr) for tr in traces]
    tok_out = [sum(s["output_tokens"] for s in tr) for tr in traces]
    n_inv = len(traces)
    cached = all(s["cached"] for tr in traces for s in tr)
    actionable_api = [sum(s["api_latency_s"] for s in o.trace) for _, o in outcomes.values()]
    cost = [(i * PRICE_IN + o * PRICE_OUT) / 1e6 for i, o in zip(tok_in, tok_out)]

    m = {
        "statuses": dict(statuses), "hold_correct": hold_correct, "hold_total": len(outcomes),
        "held": sorted(held), "draft_retries": retries,
        "natural_first_wrong": len(first_flagged), "natural_final_wrong": len(final_flagged), "natural_n": len(sent),
        "stress_n": len(stress), "stress_holds_correct": stress_holds_correct,
        "stress_sent": len(stress_sent), "stress_rules_flagged": sum(not s["rules_ok"] for s in stress_sent),
        "stress_critic_flagged": sum(not s["critic_ok"] for s in stress_sent),
        "seeded": {k: v for k, v in seeded.items()}, "seeded_missed": seeded_missed,
        "seeded_caught": sum(v[0] for v in seeded.values()), "seeded_total": sum(v[1] for v in seeded.values()),
        "rules_clean_false_flags": clean_false_flags, "critic_clean_false_flags": critic_clean_flags,
        "tone": {k: v for k, v in tone.items()},
        "avg_api_latency_per_invoice_s": sum(api_s) / n_inv,
        "avg_api_latency_per_actionable_s": sum(actionable_api) / len(actionable_api),
        "avg_tokens_in": sum(tok_in) / n_inv, "avg_tokens_out": sum(tok_out) / n_inv,
        "avg_cost_usd": sum(cost) / n_inv, "latency_source": "cache (original API latencies)" if cached else "live",
    }
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval_pipeline.json").write_text(json.dumps({**m, "stress": stress}, indent=2), encoding="utf-8")
    if verbose:
        print_report(m, stress_sent)
    return m


def print_report(m: dict, stress_sent: list[dict]) -> None:
    print(f"\n=== Pipeline (dev set, {sum(m['statuses'].values())} invoices) ===")
    print(f"outcomes: {m['statuses']}")
    print(f"hold decisions correct: {m['hold_correct']}/{m['hold_total']} actionable invoices (held: {m['held']}); "
          f"draft retries: {m['draft_retries']}")
    print(f"\n--- Verifier, natural errors ---")
    print(f"deployed (temp 0.2): drafts with a flagged value: without verifier {m['natural_first_wrong']}/"
          f"{m['natural_n']}, with verifier {m['natural_final_wrong']}/{m['natural_n']}")
    print(f"stress (temp {STRESS_TEMPERATURE}, {len(STRESS_SEEDS)} seeds): hold decisions correct "
          f"{m['stress_holds_correct']}/{m['stress_n']}; of {m['stress_sent']} sent drafts, rules flagged "
          f"{m['stress_rules_flagged']}, critic flagged {m['stress_critic_flagged']}")
    for s in stress_sent:
        if not s["rules_ok"] or not s["critic_ok"]:
            print(f"  [{s['invoice']} seed {s['seed']}] rules: {s['rules_problems'] or 'ok'} | critic: {s['critic_issues'] or 'ok'}")
    print(f"\n--- Verifier, seeded errors (deterministic rules) ---")
    for kind, (caught, total) in m["seeded"].items():
        print(f"  {kind:<32} caught {caught}/{total}")
    print(f"  {'TOTAL':<32} caught {m['seeded_caught']}/{m['seeded_total']}; "
          f"false flags on clean drafts {m['rules_clean_false_flags']}/{m['natural_n']}")
    for miss in m["seeded_missed"]:
        print(f"  MISSED {miss}")
    print(f"--- Verifier, seeded tone errors (LLM critic) ---")
    for kind, (caught, total) in m["tone"].items():
        print(f"  {kind:<36} caught {caught}/{total}")
    print(f"  false flags on clean drafts {m['critic_clean_false_flags']}/{m['natural_n']}")
    print(f"\n--- Latency and cost ({m['latency_source']}) ---")
    print(f"avg API time per invoice (all 20): {m['avg_api_latency_per_invoice_s']:.1f}s; per actionable invoice: "
          f"{m['avg_api_latency_per_actionable_s']:.1f}s")
    print(f"avg tokens per invoice: {m['avg_tokens_in']:.0f} in / {m['avg_tokens_out']:.0f} out; "
          f"≈ ${m['avg_cost_usd']:.5f} per invoice at paid-tier prices (embeddings excluded)")
