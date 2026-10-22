"""Field-level extraction accuracy against data/ground_truth.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.extract import ExtractionResult, extract_all

FIELDS = ["invoice_number", "customer_name", "customer_id", "currency", "issue_date", "due_date",
          "subtotal", "tax", "total", "line_items"]
MONEY = {"subtotal", "tax", "total"}


def _norm_text(s: str) -> str:
    return " ".join(str(s).split()).casefold()


def _money_eq(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) < 0.005


def _items_eq(pred: list[dict], truth: list[dict]) -> bool:
    if len(pred) != len(truth):
        return False
    return all(_norm_text(p["description"]) == _norm_text(t["description"]) and float(p["quantity"]) == t["quantity"]
               and _money_eq(p["unit_price"], t["unit_price"]) and _money_eq(p["amount"], t["amount"])
               for p, t in zip(pred, truth))


def field_correct(name: str, pred: dict | None, customer_id: str | None, truth: dict) -> bool:
    """Exact match after normalization (whitespace/case for text, cents for money, ISO for dates)."""
    if name == "customer_id":
        return customer_id == truth["customer_id"]
    if pred is None:
        return False
    if name in MONEY:
        return _money_eq(pred[name], truth[name])
    if name == "line_items":
        return _items_eq(pred[name], truth[name])
    if name in ("currency", "invoice_number"):
        return str(pred[name]).strip().upper() == truth[name].upper()
    return _norm_text(pred[name]) == _norm_text(truth[name])


def evaluate(results: list[ExtractionResult], truth_path: Path = Path("data/ground_truth.json")) -> dict:
    truth = {t["pdf_file"].split("/")[-1]: t for t in json.loads(truth_path.read_text(encoding="utf-8"))["invoices"]}
    per_field = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # field -> layout -> [correct, total]
    mismatches, all_correct = [], 0
    for r in results:
        t = truth[r.pdf_file]
        ok_all = True
        for f in FIELDS:
            ok = field_correct(f, r.invoice, r.customer_id, t)
            for bucket in (t["layout"], "all"):
                per_field[f][bucket][0] += ok
                per_field[f][bucket][1] += 1
            if not ok:
                ok_all = False
                got = r.customer_id if f == "customer_id" else (r.invoice or {}).get(f)
                mismatches.append({"pdf": r.pdf_file, "layout": t["layout"], "field": f,
                                   "expected": t[f], "got": got})
        all_correct += ok_all
    n = len(results)
    return {
        "n": n,
        "per_field": {f: {k: v[0] / v[1] for k, v in layouts.items()} for f, layouts in per_field.items()},
        "invoices_all_fields_correct": all_correct / n,
        "first_attempt_valid": sum(r.attempts == 1 and r.invoice is not None for r in results) / n,
        "retries_total": sum(r.attempts - 1 for r in results),
        "failed": sum(r.invoice is None for r in results),
        "avg_latency_s": sum(r.latency_s for r in results) / n,
        "avg_input_tokens": sum(r.input_tokens for r in results) / n,
        "avg_output_tokens": sum(r.output_tokens for r in results) / n,
        "avg_thought_tokens": sum(r.thought_tokens for r in results) / n,
        "all_cached": all(r.cached for r in results),
        "mismatches": mismatches,
    }


def print_report(m: dict, label: str) -> None:
    layouts = ["classic", "modern", "compact"]
    print(f"\n=== Extraction accuracy ({label}), N={m['n']} invoices ===")
    print(f"{'field':<16}{'all':>8}" + "".join(f"{l:>10}" for l in layouts))
    for f in FIELDS:
        row = m["per_field"][f]
        print(f"{f:<16}{row['all']:>8.0%}" + "".join(f"{row.get(l, 0):>10.0%}" for l in layouts))
    print(f"\ninvoices with every field correct: {m['invoices_all_fields_correct']:.0%}")
    print(f"valid JSON on first attempt: {m['first_attempt_valid']:.0%}  |  retries: {m['retries_total']}"
          f"  |  failed after retries: {m['failed']}")
    src = "from cache; latencies are from the original API calls" if m["all_cached"] else "live API calls"
    print(f"avg per invoice: {m['avg_latency_s']:.1f}s, tokens in/out/thinking = {m['avg_input_tokens']:.0f}/"
          f"{m['avg_output_tokens']:.0f}/{m['avg_thought_tokens']:.0f}  ({src})")
    for mm in m["mismatches"]:
        print(f"  MISMATCH {mm['pdf']} [{mm['layout']}] {mm['field']}: expected {mm['expected']!r}, got {mm['got']!r}")


def run(thinking_level: str | None = None, out_dir: Path = Path("outputs")) -> dict:
    results = extract_all(thinking_level=thinking_level, verbose=True)
    metrics = evaluate(results)
    label = f"thinking={thinking_level or 'default'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_extraction_{thinking_level or 'default'}.json").write_text(json.dumps(metrics, indent=2))
    print_report(metrics, label)
    return metrics
