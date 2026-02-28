from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute metrics table from minimal prediction JSONLs.")
    ap.add_argument(
        "--glob",
        type=str,
        default="grounding_eval_repro/outputs/qwen/*_predictions.jsonl",
        help="Glob for prediction jsonl files.",
    )
    ap.add_argument("--output-md", type=Path, default=None)
    return ap.parse_args()


def pct(n: float) -> str:
    return f"{n * 100:.1f}"


def acc(rows: list[dict[str, Any]], thr: float) -> float:
    if not rows:
        return 0.0
    hit = sum(1 for r in rows if float(r.get("first_iou", 0.0)) >= thr)
    return hit / len(rows)


def in_bin(d: int, name: str) -> bool:
    if name == "2-3":
        return 2 <= d <= 3
    if name == "4-6":
        return 4 <= d <= 6
    if name == ">=7":
        return d >= 7
    return False


def parse_model_variant(run_name: str, model_full_name: str, prompt_id: str) -> tuple[str, str, str]:
    p = str(prompt_id).strip().lower()
    if p in {"direct", "cot"}:
        prompt = p
    elif "_cot" in run_name or "_v2_" in run_name:
        prompt = "cot"
    else:
        prompt = "direct"
    variant = "Thinking" if "thinking" in model_full_name.lower() else "Instruct"
    model = model_full_name.replace("Qwen/", "")
    return model, variant, prompt


def main() -> None:
    args = parse_args()
    files = sorted(Path().glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    lines: list[str] = []
    lines.append("| Model | Variant | Prompt | Rows | ParseFail | Acc@0.5 | Acc@0.75 | Acc@0.9 | 2-3 | d(2-3) | 4-6 | d(4-6) | >=7 | d(>=7) | Source |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for p in files:
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            continue

        parse_fail = sum(1 for r in rows if str(r.get("parse_error", "") or "").strip())
        a50 = acc(rows, 0.5)
        a75 = acc(rows, 0.75)
        a90 = acc(rows, 0.9)

        bin_vals: dict[str, float] = {}
        for b in ["2-3", "4-6", ">=7"]:
            b_rows = [r for r in rows if in_bin(int(r.get("distractor_count", 0)), b)]
            bin_vals[b] = acc(b_rows, 0.5) if b_rows else 0.0

        run_name = str(rows[0].get("run_name", p.name.replace("_predictions.jsonl", "")))
        model_name = str(rows[0].get("model_full_name", ""))
        prompt_id = str(rows[0].get("prompt_id", ""))
        model, variant, prompt = parse_model_variant(run_name, model_name, prompt_id)

        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    variant,
                    prompt,
                    str(len(rows)),
                    str(parse_fail),
                    pct(a50),
                    pct(a75),
                    pct(a90),
                    pct(bin_vals["2-3"]),
                    f"{(bin_vals['2-3'] - a50) * 100:+.1f}",
                    pct(bin_vals["4-6"]),
                    f"{(bin_vals['4-6'] - a50) * 100:+.1f}",
                    pct(bin_vals[">=7"]),
                    f"{(bin_vals['>=7'] - a50) * 100:+.1f}",
                    f"`{p.as_posix()}`",
                ]
            )
            + " |"
        )

    out = "\n".join(lines) + "\n"
    print(out)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(out, encoding="utf-8")
        print(f"wrote={args.output_md}")


if __name__ == "__main__":
    main()
