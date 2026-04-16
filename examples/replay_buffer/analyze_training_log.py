#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import statistics
from typing import Dict, List, Tuple


KEYS_MAIN = [
    # routing & sampling
    "debug/through_route",
    "debug/replay/sample_method_code",
    "debug/batch/size",
    # rewards & masks (before KL/adv)
    "debug/scores_sum/mean",
    "debug/penalty/mean",
    "debug/response_level_rewards/mean",
    "debug/response_mask/zero_frac",
    # KL
    "debug/kl/value",
    "debug/kl/beta",
    # advantages
    "debug/response_mask_tokens",
    "debug/token_level_rewards/sum",
    "debug/advantages/sum",
    # common roll metrics (optional if present)
    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",
    "tokens/response_length/mean",
]


def try_parse_json_from_line(line: str) -> Dict:
    """Try to extract the last JSON object in a log line and parse it."""
    # Fast path: line seems to be a pure JSON
    s = line.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # Generic path: find the last {...}
    left = s.rfind("{")
    right = s.rfind("}")
    if left != -1 and right != -1 and right > left:
        frag = s[left:right + 1]
        try:
            return json.loads(frag)
        except Exception:
            return {}
    return {}


def collect_metrics(log_path: str, keys: List[str]) -> Tuple[List[Dict], Dict[str, List[float]]]:
    rows: List[Dict] = []
    series: Dict[str, List[float]] = {k: [] for k in keys}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\"system/tps\"" not in line and "\"critic/" not in line and "\"debug/" not in line and "\"tokens/" not in line:
                # quick filter to reduce parse attempts
                continue
            obj = try_parse_json_from_line(line)
            if not obj:
                continue
            row: Dict[str, float] = {}
            any_key = False
            for k in keys:
                v = obj.get(k, None)
                if isinstance(v, (int, float)):
                    row[k] = float(v)
                    series[k].append(float(v))
                    any_key = True
                else:
                    # keep alignment with NaN for csv
                    row[k] = math.nan
            if any_key:
                rows.append(row)
    return rows, series


def summarize(series: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for k, vals in series.items():
        if not vals:
            summary[k] = {"count": 0}
            continue
        valid = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        if not valid:
            summary[k] = {"count": 0}
            continue
        mean = statistics.fmean(valid)
        vmin = min(valid)
        vmax = max(valid)
        std = statistics.pstdev(valid) if len(valid) > 1 else 0.0
        last = valid[-1]
        zero_frac = sum(1 for v in valid if abs(v) < 1e-9) / len(valid)
        summary[k] = {
            "count": len(valid),
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "last": last,
            "zero_frac": zero_frac,
        }
    return summary


def write_csv(rows: List[Dict], keys: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + keys)
        for i, r in enumerate(rows):
            writer.writerow([i] + [r.get(k, math.nan) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="Path to training_*.log")
    ap.add_argument("--out_csv", default=None, help="Optional CSV output path (default: <log_dir>/analysis.csv)")
    args = ap.parse_args()

    rows, series = collect_metrics(args.log_path, KEYS_MAIN)
    summ = summarize(series)

    print("==== Metric Summary (count/mean/std/min/max/last/zero_frac) ====")
    for k in KEYS_MAIN:
        s = summ.get(k, {"count": 0})
        if s.get("count", 0) == 0:
            print(f"{k}: count=0")
            continue
        print(
            f"{k}: count={s['count']}, mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"min={s['min']:.6f}, max={s['max']:.6f}, last={s['last']:.6f}, zero_frac={s['zero_frac']:.3f}"
        )

    # Heuristic diagnostics
    print("\n==== Heuristic Diagnostics ====")
    def g(k: str, default=math.nan):
        vals = series.get(k, [])
        return vals[-1] if vals else default

    mask_zero_frac = g("debug/response_mask/zero_frac", 1.0)
    adv_sum_last = g("debug/advantages/sum", 0.0)
    rlr_mean_std = summ.get("debug/response_level_rewards/mean", {}).get("std", 0.0)
    kl_val_last = g("debug/kl/value", 0.0)
    tlr_sum_last = g("debug/token_level_rewards/sum", 0.0)

    if mask_zero_frac >= 0.99:
        print("[WARN] response_mask appears mostly zero ⇒ no learning signal over response tokens.")
    if abs(adv_sum_last) < 1e-9:
        print("[WARN] advantages sum ≈ 0 at last step ⇒ PPO gradient likely zero.")
    if rlr_mean_std < 1e-6:
        print("[WARN] response_level_rewards mean has near-zero std ⇒ rewards likely constant across batch.")
    if abs(kl_val_last) < 1e-9:
        print("[INFO] KL value near zero at last step (expected if init_kl_coef small or models aligned).")
    if abs(tlr_sum_last) < 1e-9:
        print("[WARN] token_level_rewards sum ≈ 0 ⇒ after KL penalty expansion there is little to no token reward.")

    # CSV timeline
    out_csv = args.out_csv
    if not out_csv:
        base_dir = os.path.dirname(args.log_path)
        out_csv = os.path.join(base_dir, "analysis.csv")
    write_csv(rows, KEYS_MAIN, out_csv)
    print(f"\nSaved timeline CSV: {out_csv}")


if __name__ == "__main__":
    main()


