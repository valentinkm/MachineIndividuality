#!/usr/bin/env python3
"""
Postprocessing pipeline for psychometric norm generation outputs.

Reads raw generation outputs from model_norms/, applies:
  - Multi-format file parsing (sharded, batched, offline)
  - Smart re-parsing for reasoning models (GPT-OSS, Qwen)
  - Scale validation and outlier detection
  - Completeness checking per model × norm

Outputs:
  - Cleaned CSVs in model_norms_clean/{stochastic,deterministic}/
  - Missing items lists per model × condition
  - Summary report (TXT + CSV)

Usage:
    python -m psychnorms.postprocess
"""

import datetime
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Paths (relative to project root)
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

RAW_DATA_DIR = str(PROJECT_ROOT / "outputs" / "raw_behavior" / "model_norms")
RAW_DATA_DIR_REP = str(PROJECT_ROOT / "outputs" / "raw_behavior" / "model_norms_rep")
CLEAN_DATA_DIR = str(PROJECT_ROOT / "outputs" / "raw_behavior" / "model_norms_clean")
RESOURCE_DIR = str(PROJECT_ROOT / "resources")
VOCAB_FILE = os.path.join(RESOURCE_DIR, "psychNorms_vocab.csv")
SCALE_FILE = os.path.join(RESOURCE_DIR, "norm_scales.csv")
REPORT_FILE = str(PROJECT_ROOT / "outputs" / "postprocessing_report.txt")

# Regex for parsing filenames
# Pattern 1: Sharded batched files (legacy API runs)
SHARD_PATTERN = re.compile(
    r"(.+?)_(?:vllm|hf|mock)_batched(_hitemp)?_shard(\d+)of(\d+)\.csv"
)
# Pattern 2: Non-sharded batched files (legacy API runs)
BATCHED_PATTERN = re.compile(
    r"(.+?)_(?:vllm|hf|mock)_batched(_hitemp)?\.csv"
)
# Pattern 3: DAIS offline runs
OFFLINE_PATTERN = re.compile(r"(.+?)_offline_temp(\d+\.\d+)\.csv")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")


def load_norm_scales():
    """Load validation scales from resources/norm_scales.csv."""
    scales = {}
    if not os.path.exists(SCALE_FILE):
        print(f"Warning: {SCALE_FILE} not found.")
        return scales
    try:
        df = pd.read_csv(SCALE_FILE)
        for _, row in df.iterrows():
            norm = row["norm"]
            rule = {
                "min": float(row["min_value"]) if pd.notna(row["min_value"]) else None,
                "max": float(row["max_value"]) if pd.notna(row["max_value"]) else None,
                "type": row["scale_type"],
                "valid_values": None,
            }
            if rule["type"] == "discrete" and pd.notna(row.get("valid_values")):
                try:
                    vals = {float(v.strip()) for v in str(row["valid_values"]).split(",")}
                    rule["valid_values"] = vals
                except Exception:
                    pass
            scales[norm] = rule
    except Exception as e:
        print(f"Error loading scales: {e}")
    return scales


def get_file_groups(directory):
    """
    Scan directory and group files by (model, condition).
    Condition: 'stochastic' (hitemp / temp > 0) or 'deterministic' (std / temp=0).
    """
    groups = defaultdict(list)
    files = glob.glob(os.path.join(directory, "*.csv"))

    print(f"Scanning {len(files)} files in {directory}...")

    for f in files:
        fname = os.path.basename(f)
        matched = False

        # Sharded files
        match = SHARD_PATTERN.match(fname)
        if match:
            raw_model = match.group(1).strip("_")
            is_hitemp = bool(match.group(2))
            condition = "stochastic" if is_hitemp else "deterministic"
            groups[(raw_model, condition)].append(f)
            matched = True

        # Non-sharded batched files
        if not matched:
            match = BATCHED_PATTERN.match(fname)
            if match:
                raw_model = match.group(1).strip("_")
                is_hitemp = bool(match.group(2))
                condition = "stochastic" if is_hitemp else "deterministic"
                groups[(raw_model, condition)].append(f)
                matched = True

        # DAIS offline files
        if not matched:
            match = OFFLINE_PATTERN.match(fname)
            if match:
                raw_model = match.group(1).strip("_")
                temp = float(match.group(2))
                condition = "stochastic" if temp > 0 else "deterministic"
                groups[(raw_model, condition)].append(f)
                matched = True

        # Skip known non-data files silently
        if not matched:
            if not any(x in fname for x in ["test_", "manifest", "merged", "missing"]):
                print(f"  Skipping unmatched file: {fname}")

    return groups


def count_numbers_in_response(text):
    """Count how many distinct numbers are in the raw response."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"-?\d*\.?\d+", text))


def smart_parse_rating(raw_response: str, original_rating: str) -> str:
    """
    Re-parse rating from raw_response using smarter logic for reasoning models.

    Order: assistantfinal{N} → explicit "rating: N" → last-line number →
           last number → fallback to original.
    """
    if not isinstance(raw_response, str):
        return original_rating

    raw = raw_response.strip()

    # 1. "assistantfinal{N}" or "final{N}" (GPT-OSS reasoning)
    m = re.search(r"(?:assistant)?final\s*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
    if m:
        return m.group(1)

    # 2. Explicit "rating: N" / "answer: N" etc.
    m = re.search(
        r"(?:rating|rate|answer|respond|say)[:\s]+(\d+(?:\.\d+)?)", raw, re.IGNORECASE
    )
    if m:
        return m.group(1)

    # 3. Number on the very last line
    m = re.search(r"\n\s*(\d+(?:\.\d+)?)\s*$", raw)
    if m:
        return m.group(1)

    # 4. Multiple numbers → take the last one
    all_nums = NUM_RE.findall(raw)
    if len(all_nums) > 1:
        return all_nums[-1]
    elif len(all_nums) == 1:
        return all_nums[0]

    return original_rating if original_rating else "NO_NUMBER_FOUND"


# ──────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────

def analyze_and_clean_group(model, condition, file_paths, scales, vocab_list):
    """
    Process one (model, condition) group:
      Merge → Clean → Quality metrics → Completeness check → Save.
    """
    vocab_size = len(vocab_list)
    print(f"\n--- Processing {model} [{condition}] ({len(file_paths)} files) ---")

    dfs = []
    for f in file_paths:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if not dfs:
        return None

    full_df = pd.concat(dfs, ignore_index=True)
    total_raw_rows = len(full_df)

    if "norm" not in full_df.columns or "word" not in full_df.columns:
        print("  Missing 'norm' or 'word' columns.")
        return None

    # Verbose detection
    if "raw_response" in full_df.columns:
        full_df["num_count"] = (
            full_df["raw_response"].astype(str).apply(count_numbers_in_response)
        )
        full_df["is_verbose"] = full_df["num_count"] > 1
    else:
        full_df["is_verbose"] = False

    # Rating column
    if "cleaned_rating" not in full_df.columns:
        full_df["cleaned_rating"] = np.nan
    full_df["cleaned_rating"] = full_df["cleaned_rating"].astype(str)

    # Smart re-parse for GPT-OSS (reasoning chain outputs)
    if "raw_response" in full_df.columns and "gptoss" in model.lower():
        full_df["cleaned_rating"] = full_df.apply(
            lambda row: smart_parse_rating(
                str(row.get("raw_response", "")),
                str(row.get("cleaned_rating", "")),
            ),
            axis=1,
        ).astype(str)

    # Smart re-parse for Qwen (strip <think> blocks)
    if "raw_response" in full_df.columns and "qwen" in model.lower():
        THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

        def qwen_parse(row):
            raw = str(row.get("raw_response", ""))
            original_rating = str(row.get("cleaned_rating", ""))
            cleaned_raw = THINK_RE.sub("", raw).strip()
            if cleaned_raw and cleaned_raw.replace(".", "").replace("-", "").isdigit():
                return cleaned_raw
            return smart_parse_rating(cleaned_raw, original_rating)

        full_df["cleaned_rating"] = full_df.apply(qwen_parse, axis=1).astype(str)

    full_df["cleaned_rating"] = full_df["cleaned_rating"].astype(str)

    # Error detection
    error_mask = (
        (full_df["cleaned_rating"] == "NO_NUMBER_FOUND")
        | (full_df["cleaned_rating"] == "PARSE_ERROR")
        | (full_df["cleaned_rating"].str.contains("EXCEPTION", na=False))
        | (full_df["cleaned_rating"] == "nan")
        | (full_df["cleaned_rating"] == "")
    )

    full_df["rating_val"] = pd.to_numeric(full_df["cleaned_rating"], errors="coerce")
    valid_numeric_mask = full_df["rating_val"].notna()

    # Reverse-score arousal_warriner: the prompt uses the original SAM scale
    # Recoding with 10 - val
    arousal_mask = (full_df["norm"] == "arousal_warriner") & valid_numeric_mask
    if arousal_mask.any():
        full_df.loc[arousal_mask, "rating_val"] = 10 - full_df.loc[arousal_mask, "rating_val"]
        print(f"  Recoded {arousal_mask.sum()} arousal_warriner ratings (10 - val)")

    # Combine bipolar valence: valence_mohammad = positive - negative
    # Both poles were collected as 0-3 unipolar ratings; their difference yields
    # a -3…+3 bipolar valence norm that maps to valence_mohammad in psychNorms.csv.
    # Combination is done per-repetition (matched by within-word cumcount) so the
    # LMM still sees rep-level variance across the 5 stochastic repetitions.
    # All rows with the old unipolar norm names (valid or not)
    pos_name_mask = full_df["norm"] == "valence_mohammad_positive"
    neg_name_mask = full_df["norm"] == "valence_mohammad_negative"
    # Only valid numeric rows are used for the combination
    pos_mask = pos_name_mask & valid_numeric_mask
    neg_mask = neg_name_mask & valid_numeric_mask

    if pos_mask.any() and neg_mask.any():
        pos_df = full_df[pos_mask].copy()
        neg_df = full_df[neg_mask].copy()

        # Assign within-word repetition index for alignment
        pos_df["__rep"] = pos_df.groupby("word").cumcount()
        neg_df["__rep"] = neg_df.groupby("word").cumcount()

        # Inner-merge aligns matched (word, rep) pairs; unmatched pairs are dropped
        merged = pos_df[["word", "__rep", "rating_val"]].merge(
            neg_df[["word", "__rep", "rating_val"]],
            on=["word", "__rep"],
            suffixes=("_pos", "_neg"),
        )
        merged["bipolar_val"] = merged["rating_val_pos"] - merged["rating_val_neg"]

        # Build bipolar rows using positive rows as template (preserves all metadata cols)
        bipolar_df = pos_df.merge(
            merged[["word", "__rep", "bipolar_val"]], on=["word", "__rep"]
        ).drop(columns=["__rep"])
        bipolar_df["norm"] = "valence_mohammad"
        bipolar_df["rating_val"] = bipolar_df["bipolar_val"]
        bipolar_df["cleaned_rating"] = bipolar_df["bipolar_val"].astype(str)
        bipolar_df = bipolar_df.drop(columns=["bipolar_val"])

        # Replace the two unipolar norms with the single bipolar norm
        # Use name masks (not valid-only pos_mask/neg_mask) to drop ALL unipolar rows
        full_df = pd.concat(
            [full_df[~pos_name_mask & ~neg_name_mask], bipolar_df], ignore_index=True
        )
        # Recalculate valid_numeric_mask and error_mask after modification
        valid_numeric_mask = full_df["rating_val"].notna()
        error_mask = (
            (full_df["cleaned_rating"] == "NO_NUMBER_FOUND")
            | (full_df["cleaned_rating"] == "PARSE_ERROR")
            | (full_df["cleaned_rating"].str.contains("EXCEPTION", na=False))
            | (full_df["cleaned_rating"] == "nan")
            | (full_df["cleaned_rating"] == "")
        )
        print(
            f"  Combined valence: {len(bipolar_df)} bipolar rows → norm='valence_mohammad'"
        )
    else:
        if pos_mask.any() or neg_mask.any():
            print(
                "  Warning: only one valence_mohammad pole present; skipping bipolar combination."
            )

    # Outlier detection
    full_df["is_outlier"] = False

    def check_outlier(row):
        norm = row["norm"]
        val = row["rating_val"]
        if pd.isna(val) or norm not in scales:
            return False
        rule = scales[norm]
        if rule["min"] is not None and val < rule["min"]:
            return True
        if rule["max"] is not None and val > rule["max"]:
            return True
        if rule["type"] == "discrete" and rule["valid_values"]:
            if val not in rule["valid_values"]:
                return True
        return False

    if scales:
        full_df["is_outlier"] = full_df.apply(check_outlier, axis=1)

    # Error distribution
    invalid_rows = full_df[~valid_numeric_mask]
    error_counts = (
        invalid_rows["cleaned_rating"].astype(str).value_counts().head(20).to_dict()
    )

    # Pre-dedup stats
    stats = {
        "model": model,
        "condition": condition,
        "files": len(file_paths),
        "raw_rows": total_raw_rows,
        "verbose_responses": full_df["is_verbose"].sum(),
        "parse_errors": error_mask.sum(),
        "valid_numeric": valid_numeric_mask.sum(),
        "outliers": full_df["is_outlier"].sum(),
        "error_distribution": error_counts,
    }

    # Filter: remove garbage rows (empty raw + no valid rating)
    raw_response_missing = full_df["raw_response"].isna() | (
        full_df["raw_response"].astype(str).str.strip() == ""
    )
    garbage_mask = raw_response_missing & error_mask
    final_df = full_df[~garbage_mask].copy()

    target_reps = 1 if condition == "deterministic" else 5

    # Quality scores
    qual_scores = pd.Series(2, index=final_df.index)
    final_valid = final_df["rating_val"].notna()
    final_outlier = final_df["is_outlier"]
    qual_scores[final_valid & final_outlier] = 1
    qual_scores[final_valid & ~final_outlier] = 0
    final_df["__quality_score"] = qual_scores

    # Effective validity
    base_valid = final_df["__quality_score"] == 0
    final_df["is_effective_valid"] = base_valid

    # --- CAP to target_reps valid rows per (word, norm) ---
    # For stochastic: keep first 5 effectively valid rows per (word, norm)
    # For deterministic: keep first 1 effectively valid row per (word, norm)
    # File order is preserved (earliest/zero-shot responses are kept first)
    valid_subset = final_df[final_df["is_effective_valid"]].copy()
    invalid_subset = final_df[~final_df["is_effective_valid"]].copy()

    pre_cap_valid = len(valid_subset)
    valid_subset = valid_subset.groupby(["norm", "word"]).head(target_reps)
    post_cap_valid = len(valid_subset)
    dropped_excess = pre_cap_valid - post_cap_valid

    final_df = pd.concat([valid_subset, invalid_subset], ignore_index=True)
    print(
        f"  Capped to {target_reps} valid reps/word: "
        f"{post_cap_valid:,} valid ({dropped_excess:,} excess dropped) + "
        f"{len(invalid_subset):,} invalid = {len(final_df):,} total"
    )

    # Completeness per norm
    valid_counts_series = (
        final_df[final_df["is_effective_valid"]].groupby(["norm", "word"]).size()
    )

    norm_metrics = {}
    missing_items = []

    for norm, group in final_df.groupby("norm"):
        unique_cues = group["word"].nunique()
        valid_count = group["rating_val"].notna().sum()
        outlier_count = group["is_outlier"].sum()
        error_count = (group["__quality_score"] == 2).sum()
        verbose_count = group["is_verbose"].sum()

        try:
            current_norm_counts = valid_counts_series.loc[norm]
        except KeyError:
            current_norm_counts = pd.Series(dtype=int)

        satisfied_cues_count = (current_norm_counts >= target_reps).sum()
        completeness_pct = (satisfied_cues_count / vocab_size) if vocab_size else 0
        missing_cues_count = vocab_size - satisfied_cues_count

        norm_metrics[norm] = {
            "unique_cues_found": unique_cues,
            "satisfied_cues": satisfied_cues_count,
            "missing_cues": missing_cues_count,
            "completeness_pct": completeness_pct,
            "valid_rows": valid_count,
            "error_rows": error_count,
            "outlier_rows": outlier_count,
            "verbose_rows": verbose_count,
            "mean_rating": group.loc[~group["is_outlier"], "rating_val"].mean(),
            "std_rating": group.loc[~group["is_outlier"], "rating_val"].std(),
        }

    # Missing items
    all_norms = set(final_df["norm"].unique())
    if scales:
        all_norms.update(scales.keys())
    valid_norms_str = sorted([n for n in all_norms if isinstance(n, str) and n.strip()])

    print(
        f"  Calculating missing items across {len(valid_norms_str)} norms × {vocab_size} words..."
    )

    for norm in valid_norms_str:
        try:
            norm_series = valid_counts_series.loc[norm]
        except KeyError:
            norm_series = pd.Series(dtype=int)

        for word in vocab_list:
            cur_count = norm_series.get(word, 0)
            needed = target_reps - cur_count
            if needed > 0:
                missing_items.append(
                    {
                        "word": word,
                        "norm": norm,
                        "current_valid_reps": cur_count,
                        "target_reps": target_reps,
                        "needed_reps": needed,
                    }
                )

    # Save cleaned data
    final_df["model"] = model

    # Truncate verbose GPT-OSS raw text for valid rows (saves GBs)
    if "gptoss" in model.lower() and "raw_response" in final_df.columns:
        cleanable_mask = final_df["rating_val"].notna()
        count_cleaned = cleanable_mask.sum()
        if count_cleaned > 0:
            print(
                f"   → Truncating verbose text for {count_cleaned} valid GPT-OSS rows..."
            )
            final_df.loc[cleanable_mask, "raw_response"] = "[VERBOSE_CLEANED]"

    output_cols = [c for c in final_df.columns if not c.startswith("__")]
    save_df = final_df[output_cols]

    os.makedirs(os.path.join(CLEAN_DATA_DIR, condition), exist_ok=True)
    out_filename = f"{model}_{condition}.csv"
    out_path = os.path.join(CLEAN_DATA_DIR, condition, out_filename)
    save_df.to_csv(out_path, index=False)
    print(f"  Saved cleaned data to {out_path} ({len(save_df)} rows)")

    return stats, norm_metrics, missing_items


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Starting Psych Norms Postprocessing Pipeline at {datetime.datetime.now()}")

    # Load resources
    scales = load_norm_scales()
    print(f"Loaded {len(scales)} norm scales.")

    try:
        vocab = pd.read_csv(VOCAB_FILE)
        vocab_list = vocab["word"].astype(str).tolist()
        vocab_size = len(vocab_list)
        print(f"Loaded vocab size: {vocab_size}")
    except Exception:
        print("Warning: Could not load vocab file. Completeness % will be approximate.")
        vocab_list = []
        vocab_size = 0

    # Get file groups from both raw data directories
    groups = get_file_groups(RAW_DATA_DIR)
    if os.path.isdir(RAW_DATA_DIR_REP):
        groups_rep = get_file_groups(RAW_DATA_DIR_REP)
        # Merge: for each (model, condition) key, extend the file list
        for key, files in groups_rep.items():
            groups[key].extend(files)
        print(f"Also scanned {RAW_DATA_DIR_REP} ({len(groups_rep)} additional pairs).")
    print(f"Found {len(groups)} Model × Condition pairs total.")

    csv_report_file = REPORT_FILE.replace(".txt", ".csv")
    csv_cols = [
        "model",
        "condition",
        "norm",
        "unique_cues_found",
        "satisfied_cues",
        "missing_cues",
        "completeness_pct",
        "mean_rating",
        "std_rating",
        "norm_valid_rows",
        "norm_error_rows",
        "norm_outlier_rows",
        "norm_verbose_rows",
        "input_files",
        "total_raw_rows",
        "total_verbose_failures",
        "total_parse_errors",
        "total_outliers",
    ]

    # Initialize CSV
    try:
        pd.DataFrame(columns=csv_cols).to_csv(csv_report_file, index=False)
        print(f"Initialized CSV summary: {csv_report_file}")
    except Exception as e:
        print(f"Error initializing CSV: {e}")

    with open(REPORT_FILE, "w") as report:
        report.write("PSYCH NORMS POSTPROCESSING REPORT\n")
        report.write(f"Date: {datetime.datetime.now()}\n")
        report.write("=" * 50 + "\n\n")

        for (model, condition), file_paths in sorted(groups.items()):
            res = analyze_and_clean_group(model, condition, file_paths, scales, vocab_list)
            if not res:
                report.write(
                    f"Model: {model} | Condition: {condition} → FAILED (No Data)\n\n"
                )
                continue

            stats, norm_metrics, missing_items = res

            # Append to CSV summary
            for norm, m in norm_metrics.items():
                row = {
                    "model": model,
                    "condition": condition,
                    "norm": norm,
                    "input_files": stats.get("files", 0),
                    "total_raw_rows": stats.get("raw_rows", 0),
                    "total_verbose_failures": stats.get("verbose_responses", 0),
                    "total_parse_errors": stats.get("parse_errors", 0),
                    "total_outliers": stats.get("outliers", 0),
                    "unique_cues_found": m["unique_cues_found"],
                    "satisfied_cues": m["satisfied_cues"],
                    "missing_cues": m["missing_cues"],
                    "completeness_pct": m["completeness_pct"],
                    "norm_valid_rows": m["valid_rows"],
                    "norm_error_rows": m["error_rows"],
                    "norm_outlier_rows": m["outlier_rows"],
                    "norm_verbose_rows": m["verbose_rows"],
                    "mean_rating": m["mean_rating"],
                    "std_rating": m["std_rating"],
                }
                try:
                    row_df = pd.DataFrame([row])
                    final_cols = [c for c in csv_cols if c in row_df.columns]
                    row_df[final_cols].to_csv(
                        csv_report_file, mode="a", header=False, index=False
                    )
                except Exception as e:
                    print(f"  Error appending to CSV: {e}")

            # Save missing items
            if missing_items:
                missing_df = pd.DataFrame(missing_items)
                missing_out_path = os.path.join(
                    CLEAN_DATA_DIR,
                    condition,
                    f"missing_items_{model}_{condition}.csv",
                )
                missing_df.to_csv(missing_out_path, index=False)
                print(
                    f"   → Missing items: {missing_out_path} ({len(missing_df)} tasks)"
                )
            else:
                print(f"   → No missing items for {model} {condition}! ✅")

            # Text report
            report.write(f"Model: {model} | Condition: {condition}\n")
            report.write(f"  Input Files: {stats['files']}\n")
            report.write(f"  Raw Rows: {stats['raw_rows']:,}\n")
            report.write(
                f"  Verbose Responses (>1 number): {stats['verbose_responses']:,}\n"
            )
            report.write(f"  Parse Errors: {stats['parse_errors']:,}\n")
            report.write(f"  Outliers: {stats['outliers']:,}\n")

            if "error_distribution" in stats and stats["error_distribution"]:
                report.write("\n  --- Top Unique Errors ---\n")
                for err_str, count in stats["error_distribution"].items():
                    display_str = (
                        (err_str[:75] + "..") if len(err_str) > 75 else err_str
                    )
                    report.write(f"    {display_str:<80} : {count:,}\n")

            report.write("\n  --- Norm Statistics ---\n")
            header_line = (
                f"  {'Norm':<24} | {'Found':<6} | {'Sat':<6} | {'Miss':<6} | "
                f"{'%Cmp':<6} | {'Valid':<7} | {'Err':<5} | {'Outl':<5} | "
                f"{'Verb':<5} | {'Mean':<7} | {'SD':<5}"
            )
            report.write(f"{header_line}\n")
            report.write(f"  {'-' * len(header_line)}\n")

            for norm, m in sorted(norm_metrics.items()):
                pct_str = f"{m['completeness_pct']:.1%}"
                report.write(
                    f"  {norm:<24} | {m['unique_cues_found']:<6} | "
                    f"{m['satisfied_cues']:<6} | {m['missing_cues']:<6} | "
                    f"{pct_str:<6} | {m['valid_rows']:<7} | {m['error_rows']:<5} | "
                    f"{m['outlier_rows']:<5} | {m['verbose_rows']:<5} | "
                    f"{m['mean_rating']:<7.2f} | {m['std_rating']:<5.2f}\n"
                )

            report.write("\n" + "=" * 50 + "\n\n")

    print(f"\nPipeline Complete.")
    print(f"  Report:      {REPORT_FILE}")
    print(f"  CSV Summary: {csv_report_file}")
    print(f"  Clean data:  {CLEAN_DATA_DIR}/")


if __name__ == "__main__":
    main()
