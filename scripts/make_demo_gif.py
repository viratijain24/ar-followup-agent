"""Render docs/demo.gif: a terminal REPLAY of real saved output (not a live screen recording).

Reads outputs/run_uncached.json (from `python -m src.coordinator --no-cache --out outputs/run_uncached.json`)
and outputs/eval_pipeline.json (from `python -m eval`). Run from the project root:
    python scripts/make_demo_gif.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, PAD, LINE_H, COLS = 960, 560, 18, 19, 94
BG, FG, DIM, GREEN, YELLOW, CYAN, RED = "#1e1e1e", "#d4d4d4", "#808080", "#6a9955", "#dcdcaa", "#4fc1ff", "#f48771"
FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
BOLD = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 15)
STATUS_COLOR = {"ready_to_send": GREEN, "needs_human_review": YELLOW, "held": CYAN, "flagged": RED}


def render(lines: list[tuple[str, str]], title: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 26], fill="#333333")
    d.text((PAD, 5), title, fill=DIM, font=FONT)
    label = "replay of real saved output - not a live recording"
    d.text((W - PAD - d.textlength(label, font=FONT), 5), label, fill=DIM, font=FONT)
    visible = lines[-((H - 40) // LINE_H):]  # scroll like a terminal
    for i, (text, color) in enumerate(visible):
        d.text((PAD, 34 + i * LINE_H), text, fill=color, font=BOLD if text.startswith("$") else FONT)
    return img


def main() -> None:
    run = json.loads(Path("outputs/run_uncached.json").read_text(encoding="utf-8"))
    pipe = json.loads(Path("outputs/eval_pipeline.json").read_text(encoding="utf-8"))
    frames, durations, lines = [], [], []

    def show(ms: int, title: str = "PowerShell - ar-followup-agent") -> None:
        frames.append(render(lines, title))
        durations.append(ms)

    def type_cmd(cmd: str) -> None:
        lines.append(("$ ", FG))
        for i in range(0, len(cmd) + 1, 6):
            lines[-1] = (f"$ {cmd[:i]}", FG)
            show(60)
        lines[-1] = (f"$ {cmd}", FG)
        show(500)

    type_cmd("python -m src.coordinator --no-cache")
    lines.append(("# 20/20 invoices extracted (Gemini reads each PDF -> validated JSON); 11 need no action", DIM))
    show(900)
    for o in run:
        if o["status"] == "no_action":
            continue
        reason = f" ({o['reason'][:58]}...)" if len(o["reason"]) > 60 else (f" ({o['reason']})" if o["reason"] else "")
        lines.append((f"{o['invoice_number']} [{o['stage']}] -> {o['status']}{reason}", STATUS_COLOR.get(o["status"], FG)))
        show(450)
    counts: dict[str, int] = {}
    for o in run:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    lines.append((f"{counts}", FG))
    show(1800)

    lines.clear()
    lines.append(("# Draft for INV-1008 (35 days overdue, partly paid), from outputs/run_uncached.json", DIM))
    lines.append(("", FG))
    draft = next(o for o in run if o["invoice_number"] == "INV-1008")["draft"]
    lines.append((f"Subject: {draft['subject']}", CYAN))
    for para in draft["body"].split("\n"):
        for wrapped in textwrap.wrap(para, COLS) or [""]:
            lines.append((wrapped, FG))
    show(6500)

    lines.clear()
    lines.append(("# Selected results from `python -m eval` (outputs/eval_pipeline.json)", DIM))
    lines.append(("", FG))
    show(800)
    seeded = pipe["seeded"]
    rows = [("Verifier, seeded value errors caught:", f"{pipe['seeded_caught']}/{pipe['seeded_total']}")]
    rows += [(f"  {k}", f"{v[0]}/{v[1]}") for k, v in seeded.items()]
    rows += [("Rules false flags on clean drafts:", f"{pipe['rules_clean_false_flags']}/{pipe['natural_n']}"),
             ("LLM critic, seeded tone errors caught:",
              f"{sum(v[0] for v in pipe['tone'].values())}/{sum(v[1] for v in pipe['tone'].values())}"),
             ("Hold decisions correct (deployed / stress):",
              f"{pipe['hold_correct']}/{pipe['hold_total']} / {pipe['stress_holds_correct']}/{pipe['stress_n']}"),
             ("API time per invoice:", f"{pipe['avg_api_latency_per_invoice_s']:.1f}s"),
             ("Approx. cost per invoice:", f"${pipe['avg_cost_usd']:.4f}")]
    for name, value in rows:
        lines.append((f"{name:<48}{value}", GREEN if name.startswith("  ") else FG))
        show(350)
    show(5000)

    out = Path("docs/demo.gif")
    out.parent.mkdir(exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
