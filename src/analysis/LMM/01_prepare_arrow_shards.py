#!/usr/bin/env python3
"""
01_prepare_arrow_shards.py
Pre-processor to shard massive clean CSV files into tightly compressed,
norm-specific Apache Arrow (.parquet) files for extreme I/O performance in R.

WORD FILTERING: For each norm, only words that have at least 1 valid
response from EVERY model are retained. This enforces a fully crossed
design for the downstream LMM variance partitioning.

Parallelized for high-core servers (128-core, 500GB+ RAM):
  - Parallel CSV reading via multiprocessing
  - Parallel per-norm Parquet writing

Usage:
    python3 src/analysis/LMM/01_prepare_arrow_shards.py
    python3 src/analysis/LMM/01_prepare_arrow_shards.py --max-words 1000 --norms concreteness_brysbaert arousal_warriner
"""

import os
import glob
import time
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("❌ Missing required libraries. Please run: pip install pandas pyarrow")
    exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

INPUT_DIR = str(PROJECT_ROOT / "outputs" / "raw_behavior" / "model_norms_clean" / "stochastic")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "raw_behavior" / "model_norms_arrow")

REQUIRED_COLS = ['model', 'norm', 'word', 'rating_val', 'is_effective_valid']


def _read_one_csv(fpath):
    """Read a single CSV, filter columns, return DataFrame."""
    try:
        df = pd.read_csv(fpath, usecols=REQUIRED_COLS, low_memory=False)
        print(f"  ✅ {os.path.basename(fpath)}: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  ⚠️ Error reading {fpath}: {e}")
        return pd.DataFrame()


def _write_one_shard(args):
    """Write a single norm shard to Parquet."""
    norm, norm_df, output_dir = args
    norm_df = norm_df.astype({
        'model': 'category',
        'norm': 'category',
        'word': 'category',
        'rating_val': 'float32'
    })
    outfile = os.path.join(output_dir, f"{norm}.parquet")
    table = pa.Table.from_pandas(norm_df, preserve_index=False)
    pq.write_table(table, outfile, compression='snappy')
    return norm, len(norm_df)


def _filter_fully_crossed_words(norm_df, norm_name, all_models):
    """
    Filter to only words that have ≥1 valid response from EVERY model.

    Args:
        norm_df: DataFrame with valid rows for a single norm
        norm_name: Name of the norm (for logging)
        all_models: Set of all model IDs that must be present

    Returns:
        Filtered DataFrame containing only fully-crossed words
    """
    n_models = len(all_models)
    n_words_before = norm_df['word'].nunique()

    # For each word, find the set of models that contributed ≥1 valid row
    word_model_coverage = norm_df.groupby('word')['model'].nunique()

    # Keep only words covered by ALL models
    fully_crossed_words = set(word_model_coverage[word_model_coverage == n_models].index)
    n_words_after = len(fully_crossed_words)
    n_words_dropped = n_words_before - n_words_after

    # Filter
    filtered = norm_df[norm_df['word'].isin(fully_crossed_words)]
    rows_before = len(norm_df)
    rows_after = len(filtered)

    print(f"  {norm_name:35s}  words: {n_words_before:>7,} → {n_words_after:>7,}  "
          f"(−{n_words_dropped:,}, {n_words_after/n_words_before:.4%} retained)  "
          f"rows: {rows_before:>9,} → {rows_after:>9,}")

    return filtered


def build_parquet_shards(max_words=None, norm_filter=None):
    print("============================================================")
    print("ARROW SHARDING OPTIMIZER: Prepivoting Data by Norm")
    print("  ≥1-rep word filtering: fully crossed design enforcement")
    if max_words:
        print(f"  ⚡ TEST MODE: max {max_words} words per norm")
    if norm_filter:
        print(f"  ⚡ TEST MODE: norms limited to {norm_filter}")
    print("============================================================")

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_stochastic.csv")))
    files = [f for f in files if "missing_items" not in os.path.basename(f)]

    if not files:
        print("❌ No stochastic CSVs found to shard.")
        return

    n_workers = min(len(files), cpu_count())
    print(f"Loading {len(files)} CSV components ({n_workers} workers)")
    start_time = time.time()

    # Parallel CSV reading
    with Pool(n_workers) as pool:
        chunks = pool.map(_read_one_csv, files)

    chunks = [c for c in chunks if len(c) > 0]
    full_df = pd.concat(chunks, ignore_index=True)
    del chunks  # free memory
    load_time = time.time() - start_time
    print(f"✅ Loaded {len(full_df):,} total rows in {load_time:.1f}s.")

    print("\nApplying LMM filters (is_effective_valid == True & rating_val.notna())...")
    full_df['rating_val'] = pd.to_numeric(full_df['rating_val'], errors='coerce')

    valid_mask = full_df['rating_val'].notna()
    if 'is_effective_valid' in full_df.columns:
        valid_mask = valid_mask & (full_df['is_effective_valid'] == True)

    filtered_df = full_df[valid_mask].copy()
    dropped_rows = len(full_df) - len(filtered_df)
    del full_df  # free memory
    print(f"Filtered valid rows: {len(filtered_df):,} | Dropped: {dropped_rows:,}")

    # Filter to requested norms (test mode)
    if norm_filter:
        before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['norm'].isin(norm_filter)]
        print(f"Norm filter applied: {before:,} → {len(filtered_df):,} rows "
              f"({len(norm_filter)} norms: {norm_filter})")

    # Identify the global set of models (should be 10)
    all_models = set(filtered_df['model'].unique())
    print(f"\nModels detected: {len(all_models)} — {sorted(all_models)}")

    # Per-norm word filtering: keep only fully crossed words
    print(f"\n── Per-norm word filtering (≥1 valid rep from all {len(all_models)} models) ──")
    unique_norms = sorted(filtered_df['norm'].unique())

    shard_args = []
    total_words_dropped = 0
    total_rows_before = 0
    total_rows_after = 0

    for norm in unique_norms:
        norm_df = filtered_df[filtered_df['norm'] == norm]
        total_rows_before += len(norm_df)
        n_words_before = norm_df['word'].nunique()

        norm_df_filtered = _filter_fully_crossed_words(norm_df, norm, all_models)

        # Apply max-words limit (test mode): deterministic subset of fully crossed words
        if max_words and norm_df_filtered['word'].nunique() > max_words:
            keep_words = sorted(norm_df_filtered['word'].unique())[:max_words]
            norm_df_filtered = norm_df_filtered[norm_df_filtered['word'].isin(keep_words)]
            print(f"    → Test mode: capped to {max_words} words "
                  f"({len(norm_df_filtered):,} rows)")

        total_rows_after += len(norm_df_filtered)
        n_words_after = norm_df_filtered['word'].nunique()
        total_words_dropped += (n_words_before - n_words_after)

        shard_args.append((norm, norm_df_filtered, OUTPUT_DIR))

    del filtered_df  # free memory

    print(f"\n  TOTAL: {total_words_dropped:,} word-norm exclusions across {len(unique_norms)} norms")
    print(f"  TOTAL rows: {total_rows_before:,} → {total_rows_after:,} "
          f"({total_rows_after/total_rows_before:.4%} retained)")

    # Parallel Parquet writing
    print(f"\nWriting {len(unique_norms)} Parquet shards...")

    with Pool(len(unique_norms)) as pool:
        results = pool.map(_write_one_shard, shard_args)

    for norm, n_rows in sorted(results):
        print(f"  → Saved {norm}.parquet ({n_rows:,} rows)")

    total_time = time.time() - start_time
    print("\n============================================================")
    print(f"✅ Sharding Complete in {total_time:.1f}s.")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arrow shard builder")
    parser.add_argument("--max-words", type=int, default=None,
                        help="Limit to N words per norm (test mode)")
    parser.add_argument("--norms", nargs="+", default=None,
                        help="Only shard these norms (test mode)")
    args = parser.parse_args()

    build_parquet_shards(max_words=args.max_words, norm_filter=args.norms)
