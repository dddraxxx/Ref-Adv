from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import Image, load_dataset
import requests


DEFAULT_FOLLOWUP = (
    "Your previous answer did not end with a valid JSON bbox output.\n"
    "Now output ONLY one complete JSON code block.\n"
    "Use this exact schema:\n"
    "```json\n"
    '{"bboxes": [[x1, y1, x2, y2]]}\n'
    "```\n"
    "Do not output any extra text."
)

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.S | re.I)


@dataclass
class ModelSpec:
    key: str
    model_full_name: str
    pred_box_expected_format: str


@dataclass
class RunSpec:
    run_name: str
    model_key: str
    prompt_id: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal Qwen grounding eval runner.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-name", type=str, required=True)
    return ap.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def image_to_jpeg_bytes(image_obj: Any) -> bytes:
    rgb = image_obj.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def build_user_message(image_bytes: bytes, prompt: str) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
    }


def extract_assistant_text(raw_json: dict[str, Any]) -> str:
    try:
        choices = raw_json.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            return "\n".join(p for p in parts if p)
        return str(content)
    except Exception:
        return ""


def finish_reason(raw_json: dict[str, Any]) -> str:
    try:
        choices = raw_json.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("finish_reason", "") or "").strip().lower()
    except Exception:
        return ""


def post_chat_completion(
    *,
    base_url: str,
    model_full_name: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout_sec: int,
) -> tuple[dict[str, Any], float]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_full_name,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": int(seed),
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return r.json(), time.time() - t0


def _load_last_json_candidate(text: str) -> Any | None:
    fenced_chunks = [m.group(1).strip() for m in FENCE_RE.finditer(text) if m.group(1).strip()]
    for c in reversed(fenced_chunks):
        try:
            return json.loads(c)
        except Exception:
            continue

    t = text.rstrip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch not in "[{":
            continue
        try:
            obj, end = decoder.raw_decode(t[i:])
            if i + end == len(t):
                return obj
        except Exception:
            continue
    return None


def _collect_boxes(obj: Any, out: list[list[float]], *, max_boxes: int = 100) -> None:
    if len(out) >= max_boxes:
        return
    if isinstance(obj, list):
        if len(obj) == 4 and all(isinstance(x, (int, float)) for x in obj):
            out.append([float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])])
            return
        for it in obj:
            _collect_boxes(it, out, max_boxes=max_boxes)
        return
    if isinstance(obj, dict):
        priority_keys = ["bboxes", "boxes", "bbox", "bbox_2d", "predictions", "results", "objects", "coordinates"]
        for k in priority_keys:
            if k in obj:
                _collect_boxes(obj[k], out, max_boxes=max_boxes)
        for k, v in obj.items():
            if k in priority_keys:
                continue
            _collect_boxes(v, out, max_boxes=max_boxes)


def to_abs_xyxy(box: list[float], *, width: int, height: int, fmt: str) -> list[float]:
    x1, y1, x2, y2 = box
    if fmt == "abs_xyxy":
        pass
    elif fmt == "norm_1000_xyxy":
        x1, y1, x2, y2 = x1 / 1000.0 * width, y1 / 1000.0 * height, x2 / 1000.0 * width, y2 / 1000.0 * height
    elif fmt == "norm_1_xyxy":
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    else:
        raise ValueError(f"Unsupported pred_box_expected_format: {fmt}")

    xa, xb = sorted([float(x1), float(x2)])
    ya, yb = sorted([float(y1), float(y2)])
    xa = min(max(xa, 0.0), float(width))
    xb = min(max(xb, 0.0), float(width))
    ya = min(max(ya, 0.0), float(height))
    yb = min(max(yb, 0.0), float(height))
    return [xa, ya, xb, yb]


def parse_qwen_bboxes(text: str, *, width: int, height: int, fmt: str) -> tuple[list[list[float]], str]:
    if not text or not text.strip():
        return [], "empty_response"
    parsed = _load_last_json_candidate(text)
    if parsed is None:
        return [], "no_bbox_found"

    raw_boxes: list[list[float]] = []
    _collect_boxes(parsed, raw_boxes)
    if not raw_boxes:
        return [], "no_bbox_found"

    out: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for b in raw_boxes:
        bb = to_abs_xyxy(b, width=width, height=height, fmt=fmt)
        key = (round(bb[0], 6), round(bb[1], 6), round(bb[2], 6), round(bb[3], 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(bb)
    return out, ""


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def parse_distractor_count(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, int):
        return max(0, int(x))
    if isinstance(x, list):
        return len(x)
    s = str(x).strip()
    if not s:
        return 0
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, int):
            return max(0, int(obj))
    except Exception:
        pass
    nums = re.findall(r"\d+", s)
    if nums:
        return len(nums)
    return 0


def select_rows(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    dcfg = cfg["dataset"]
    token = os.environ.get(str(dcfg.get("token_env", "HF_TOKEN")), "").strip() or None
    ds = load_dataset(str(dcfg["repo_id"]), split=str(dcfg.get("split", "train")), token=token)
    ds = ds.cast_column("image", Image(decode=True))
    total_rows = len(ds)

    start = int(dcfg.get("start_index", 0))
    limit = int(dcfg.get("limit", 0))
    end = total_rows if limit <= 0 else min(total_rows, start + limit)

    ff = dcfg.get("filter", {}) or {}
    f_human = ff.get("human_authored")
    f_source = ff.get("image_source")

    rows: list[dict[str, Any]] = []
    for i in range(start, end):
        r = ds[i]
        if f_human is not None and bool(r.get("human_authored")) != bool(f_human):
            continue
        if f_source is not None and str(r.get("image_source", "")) != str(f_source):
            continue
        rows.append(
            {
                "row_idx": i,
                "image": r["image"],
                "file_name": str(r.get("file_name", "")),
                "normal_caption": str(r.get("normal_caption", "")),
                "solution": [float(v) for v in r.get("solution", [])],
                "width": int(r.get("width", 0)),
                "height": int(r.get("height", 0)),
                "image_source": str(r.get("image_source", "")),
                "human_authored": bool(r.get("human_authored", False)),
                "use_negation": bool(r.get("use_negation", False)),
                "distractor_count": parse_distractor_count(r.get("distractors")),
            }
        )
    return rows, total_rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config.resolve())

    models = {
        k: ModelSpec(
            key=k,
            model_full_name=str(v["model_full_name"]),
            pred_box_expected_format=str(v["pred_box_expected_format"]),
        )
        for k, v in (cfg.get("models", {}) or {}).items()
    }

    runs: dict[str, RunSpec] = {}
    for r in (cfg.get("runs", []) or []):
        rs = RunSpec(run_name=str(r["run_name"]), model_key=str(r["model_key"]), prompt_id=str(r["prompt_id"]))
        runs[rs.run_name] = rs

    if args.run_name not in runs:
        raise SystemExit(f"run_name not found: {args.run_name}")
    run = runs[args.run_name]

    if run.model_key not in models:
        raise SystemExit(f"model_key not found: {run.model_key}")
    model = models[run.model_key]

    prompts = cfg.get("prompts", {}) or {}
    if run.prompt_id not in prompts:
        raise SystemExit(f"prompt_id not found: {run.prompt_id}")
    prompt_template = str(prompts[run.prompt_id])

    out_dir = Path(str(cfg["output"]["dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run.run_name}_predictions.jsonl"

    base_url = str(cfg["server"]["base_url"])
    gen_cfg = cfg.get("generation", {}) or {}
    max_tokens = int(gen_cfg.get("max_tokens", 1024))
    temperature = float(gen_cfg.get("temperature", 0.0))
    top_p = float(gen_cfg.get("top_p", 1.0))
    seed = int(gen_cfg.get("seed", 7))
    timeout_sec = int(gen_cfg.get("timeout_sec", 180))

    eval_cfg = cfg.get("evaluation", {}) or {}
    iou_threshold = float(eval_cfg.get("iou_threshold", 0.5))
    retry_followup_on_parse_fail = bool(eval_cfg.get("retry_followup_on_parse_fail", True))
    retry_followup_length_only = bool(eval_cfg.get("retry_followup_length_only", False))
    retry_followup_text = str(eval_cfg.get("retry_followup_text", DEFAULT_FOLLOWUP))
    retry_followup_max_tokens = eval_cfg.get("retry_followup_max_tokens")
    retry_followup_max_tokens = int(retry_followup_max_tokens) if retry_followup_max_tokens is not None else max_tokens

    rows, total_rows = select_rows(cfg)
    print(f"dataset_total_rows={total_rows}")
    print(f"rows_selected={len(rows)}")
    print(f"run_name={run.run_name}")
    print(f"model_full_name={model.model_full_name}")

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            prompt = prompt_template.format(ref_sentence=row["normal_caption"])
            image_bytes = image_to_jpeg_bytes(row["image"])

            parse_error = ""
            pred_boxes: list[list[float]] = []
            retry_used = False

            try:
                pass1_raw, _ = post_chat_completion(
                    base_url=base_url,
                    model_full_name=model.model_full_name,
                    messages=[build_user_message(image_bytes, prompt)],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    timeout_sec=timeout_sec,
                )
                pass1_text = extract_assistant_text(pass1_raw)
                pass1_finish = finish_reason(pass1_raw)
                pred_boxes, parse_error = parse_qwen_bboxes(
                    pass1_text,
                    width=row["width"],
                    height=row["height"],
                    fmt=model.pred_box_expected_format,
                )

                can_retry = retry_followup_on_parse_fail and bool(parse_error)
                if can_retry and retry_followup_length_only and pass1_finish != "length":
                    can_retry = False

                if can_retry:
                    retry_used = True
                    pass2_raw, _ = post_chat_completion(
                        base_url=base_url,
                        model_full_name=model.model_full_name,
                        messages=[
                            build_user_message(image_bytes, prompt),
                            {"role": "assistant", "content": pass1_text},
                            {"role": "user", "content": retry_followup_text},
                        ],
                        max_tokens=retry_followup_max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                        timeout_sec=timeout_sec,
                    )
                    pass2_text = extract_assistant_text(pass2_raw)
                    pred_boxes, parse_error = parse_qwen_bboxes(
                        pass2_text,
                        width=row["width"],
                        height=row["height"],
                        fmt=model.pred_box_expected_format,
                    )
            except Exception:
                parse_error = "backend_error"

            pred_first = pred_boxes[0] if pred_boxes else []
            gt = row["solution"]
            first_iou = iou_xyxy(gt, pred_first) if len(gt) == 4 and len(pred_first) == 4 else 0.0

            out = {
                "row_idx": row["row_idx"],
                "file_name": row["file_name"],
                "normal_caption": row["normal_caption"],
                "image_source": row["image_source"],
                "human_authored": row["human_authored"],
                "use_negation": row["use_negation"],
                "distractor_count": row["distractor_count"],
                "gt_bbox_xyxy": gt,
                "pred_box_xyxy_first": pred_first,
                "first_iou": first_iou,
                "first_hit": first_iou >= iou_threshold,
                "parse_error": parse_error,
                "retry_followup_used": retry_used,
                "model_full_name": model.model_full_name,
                "prompt_id": run.prompt_id,
                "pred_box_expected_format": model.pred_box_expected_format,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

            if i == 1 or i % 50 == 0 or i == len(rows):
                print(f"progress={i}/{len(rows)} parse_error={parse_error or 'ok'}")

    print(f"predictions_jsonl={out_path}")


if __name__ == "__main__":
    main()
