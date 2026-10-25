"""Run the evaluation suite:  python -m eval [--no-cache] [--data data|data_holdout]

Sections: extraction accuracy, retrieval, end-to-end pipeline (holds, verifier, latency/cost),
then one combined results table. Retrieval and pipeline sections need the dev set's corpus labels,
so they run only for --data data.
"""
import argparse
from pathlib import Path

from eval import extraction, pipeline, retrieval
from src import llm


def summary(ext: dict, ret: dict, pipe: dict) -> None:
    rows = [
        ("Extraction: invoices with every field correct", f"{ext['invoices_all_fields_correct']:.0%} of {ext['n']}"),
        ("Extraction: lowest per-field accuracy", f"{min(f['all'] for f in ext['per_field'].values()):.0%}"),
        ("Retrieval (hybrid): relevant emails in top 1 / top 2",
         f"{ret['recall']['hybrid@1_all']:.0%} / {ret['recall']['hybrid@2_all']:.0%} of {ret['relevant']['all']} "
         f"(max {ret['ceiling'][1]} / {ret['ceiling'][2]})"),
        ("Hold decisions correct (deployed / stress sample)",
         f"{pipe['hold_correct']}/{pipe['hold_total']} / {pipe['stress_holds_correct']}/{pipe['stress_n']}"),
        ("Drafts with a wrong value, no verifier (deployed / stress)",
         f"{pipe['natural_first_wrong']}/{pipe['natural_n']} / {pipe['stress_rules_flagged']}/{pipe['stress_sent']}"),
        ("Drafts with a wrong value, with verifier (deployed)", f"{pipe['natural_final_wrong']}/{pipe['natural_n']}"),
        ("Seeded value errors caught by rules", f"{pipe['seeded_caught']}/{pipe['seeded_total']}"),
        ("Rules false flags on clean drafts", f"{pipe['rules_clean_false_flags']}/{pipe['natural_n']}"),
        ("Seeded tone errors caught by LLM critic",
         f"{sum(v[0] for v in pipe['tone'].values())}/{sum(v[1] for v in pipe['tone'].values())}"),
        ("Critic flags on clean drafts (deployed / stress)",
         f"{pipe['critic_clean_false_flags']}/{pipe['natural_n']} / {pipe['stress_critic_flagged']}/{pipe['stress_sent']}"),
        ("Avg API time per invoice (all / actionable)",
         f"{pipe['avg_api_latency_per_invoice_s']:.1f}s / {pipe['avg_api_latency_per_actionable_s']:.1f}s"),
        ("Avg tokens per invoice (in / out)", f"{pipe['avg_tokens_in']:.0f} / {pipe['avg_tokens_out']:.0f}"),
        ("Approx. cost per invoice (paid-tier prices)", f"${pipe['avg_cost_usd']:.5f}"),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"\n{'=' * (width + 30)}\nRESULTS (dev set, seed 42, {llm.chat_model()})\n{'=' * (width + 30)}")
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses (fresh timing run)")
    parser.add_argument("--thinking", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="dataset dir (data = dev set, seed 42; data_holdout = held-out set, seed 7)")
    args = parser.parse_args()
    llm.set_cache(not args.no_cache, embeddings=True)
    ext = extraction.run(args.data, thinking_level=args.thinking)
    if args.data == Path("data"):  # retrieval labels and expected holds are written for the dev set's corpus
        ret = retrieval.run(args.data)
        pipe = pipeline.run(args.data)
        summary(ext, ret, pipe)


if __name__ == "__main__":
    main()
