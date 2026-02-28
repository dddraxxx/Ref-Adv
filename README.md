# Grounding Eval Repro

Minimal public repro package for Ref-Adv grounding eval (Qwen series, temperature=0.0 setting).

## Scope

This directory contains:

- Minimal eval code (`run.py`, `report.py`)
- One consolidated config (`configs/qwen.yaml`)
- Minimal prediction JSONLs for all completed Qwen runs (temp=0.0)

This package intentionally does **not** include summary JSON files.

## Install

Run commands from this directory.

```bash
python -m pip install -r requirements.txt
```

## Run One Eval

Start a vLLM OpenAI-compatible server first (model must match the run's `model_full_name`).

Then run:

```bash
python run.py \
  --config configs/qwen.yaml \
  --run-name qwen35a35b_direct
```

Output:

- `outputs/qwen/<run_name>_predictions.jsonl`

## Compute Metrics Table

```bash
python report.py \
  --glob 'outputs/qwen/*_predictions.jsonl' \
  --output-md '../doc/eval_table.md'
```

The report includes:

- `Acc@0.5`, `Acc@0.75`, `Acc@0.9`
- parse-fail count
- distractor bins (`2-3`, `4-6`, `>=7`) and delta vs overall `Acc@0.5`

## Minimal JSONL Row Schema

Each line contains only:

- `row_idx`
- `file_name`
- `normal_caption`
- `image_source`
- `human_authored`
- `use_negation`
- `distractor_count`
- `gt_bbox_xyxy`
- `pred_box_xyxy_first`
- `first_iou`
- `first_hit`
- `parse_error`
- `retry_followup_used`
- `model_full_name`
- `prompt_id`
- `pred_box_expected_format`

## Notes

- Prompt IDs are `direct` and `cot`.
- Decode setting is fixed in config (`temperature=0.0`, `top_p=1.0`).
- IoU uses absolute `xyxy` (`solution` from HF dataset) as GT.
- Prediction coordinate format is fixed per model in config (`abs_xyxy` or `norm_1000_xyxy`).
- Parse-fail rows are counted in denominator for accuracy metrics.
