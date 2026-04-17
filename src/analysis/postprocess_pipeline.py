#!/usr/bin/env python3
"""
03_postprocess_pipeline.py — Data Postprocessing (Pipeline Step 0)

Merges and cleans raw model norm CSVs into analysis-ready datasets.

Operations:
  1. Merge per-model raw CSV files (handles sharded, full-run, and offline formats)
  2. Parse and validate ratings against norm-specific scales (resources/norm_scales.csv)
  3. Combine bipolar valence (positive - negative) into a single valence_mohammad score
  4. Cap stochastic repetitions at 5 per (model, norm, word)
  5. Flag outliers (out-of-scale ratings) and compute is_effective_valid
  6. Split into stochastic/ (T=1.0) and deterministic/ (T=0) subdirectories

Inputs:  outputs/raw_behavior/model_norms/*.csv
Outputs: outputs/raw_behavior/model_norms_clean/{stochastic,deterministic}/*.csv
Report:  outputs/postprocessing_report.txt

Usage:
    python src/analysis/03_postprocess_pipeline.py
    # or via module: python -m psychnorms.postprocess
"""

import pandas as pd
import glob
import os
import re
import sys
import csv as csv_mod
import numpy as np
from collections import defaultdict
import datetime

# --- Configuration ---
RAW_DATA_DIR = "outputs/raw_behavior/model_norms"
CLEAN_DATA_DIR = "outputs/raw_behavior/model_norms_clean"
RESOURCE_DIR = "resources"
VOCAB_FILE = os.path.join(RESOURCE_DIR, "psychNorms_vocab.csv")
SCALE_FILE = os.path.join(RESOURCE_DIR, "norm_scales.csv")
REPORT_FILE = "outputs/postprocessing_report.txt"

# Regex for parsing filenames - multiple patterns for different source types
# Pattern 1: Sharded batched files (legacy API runs)
# Examples: falcon_h1_34b_it_vllm_batched_hitemp_shard0of1.csv
SHARD_PATTERN = re.compile(r"(.+?)_(?:vllm|hf|mock)_batched(_hitemp)?_shard(\d+)of(\d+)\.csv")

# Pattern 2: Non-sharded batched files (legacy API runs)
# Examples: falcon_h1_34b_it_vllm_batched.csv, falcon_h1_34b_it_vllm_batched_hitemp.csv
BATCHED_PATTERN = re.compile(r"(.+?)_(?:vllm|hf|mock)_batched(_hitemp)?\.csv")

# Pattern 3: DAIS offline runs
# Examples: falcon_h1_34b_it_offline_temp0.0.csv, phi_4_offline_temp1.0.csv
OFFLINE_PATTERN = re.compile(r"(.+?)_offline_temp(\d+\.\d+)\.csv")

# Attempt type mapping: offline files store as integer 0-4, batched files as strings
ATTEMPT_CODE_MAP = {
    '0': 'zero_shot', '1': 'retry_s1_scale', '2': 'retry_s2_parse',
    '3': 'retry_s3_temp', '4': 'retry_s4_refusal',
}
ATTEMPT_CODE_MAP_FLOAT = {float(k): v for k, v in ATTEMPT_CODE_MAP.items()}


def read_raw_csv_robust(filepath):
    """
    Read a raw CSV file, recovering rows silently dropped by pd.read_csv.

    Problem: offline generation files sometimes have rows with an extra
    attempt_type column (integer 0-4) appended mid-run without updating
    the 9-column header.  Standard pd.read_csv drops these rows because
    they have more fields than the header declares.

    Additionally, some raw_response fields contain unquoted commas,
    producing 11-field rows that are also dropped.

    Strategy:
      Pass 1 — pd.read_csv with a 10-column schema to capture both
               9-field rows (attempt_type = NaN → 'zero_shot') and
               10-field rows (attempt_type = 0-4 → mapped to label).
      Pass 2 — csv.reader scan to reconstruct 11+-field rows by
               rejoining the split raw_response from the middle fields.
    """
    # Read header to determine schema
    with open(filepath) as f:
        reader = csv_mod.reader(f)
        header = next(reader)

    # If attempt_type is already in the header (batched/API files), read normally
    if 'attempt_type' in header:
        return pd.read_csv(filepath, low_memory=False)

    n_cols = len(header)  # expected: 9

    # --- Pass 1: pandas with extended 10-column schema ---
    extended_header = header + ['_attempt_code']
    df = pd.read_csv(
        filepath,
        names=extended_header,
        header=None,        # we supply our own names
        skiprows=1,         # skip the original header row
        on_bad_lines='skip',
        low_memory=False,
    )
    n_pass1 = len(df)

    # Map numeric attempt codes to string labels; NaN (9-field rows) → zero_shot
    df['attempt_type'] = df['_attempt_code'].map(ATTEMPT_CODE_MAP_FLOAT).fillna('zero_shot')
    df.drop(columns=['_attempt_code'], inplace=True)

    # --- Quick check: any rows skipped? ---
    with open(filepath, 'rb') as f:
        n_file_lines = sum(1 for _ in f) - 1  # physical lines minus header
    if n_file_lines <= n_pass1:
        return df  # all rows captured

    # --- Pass 2: recover rows with >10 fields via csv.reader ---
    recovered = []
    with open(filepath) as f:
        reader = csv_mod.reader(f)
        next(reader)  # skip header
        for row in reader:
            n = len(row)
            if n <= n_cols + 1:
                continue  # already captured by Pass 1

            # Determine tail structure
            has_attempt = row[-1] in ATTEMPT_CODE_MAP
            n_tail = 4 if has_attempt else 3  # cleaned_text, cleaned_rating, temp[, attempt]
            n_prefix = 5  # model_key, backend, endpoint_url, norm, word

            if n < n_prefix + n_tail + 1:
                continue  # too short to reconstruct

            # Reconstruct: prefix is clean, raw_response is middle, tail is clean
            data = {header[i]: row[i] for i in range(n_prefix)}
            data[header[5]] = ','.join(row[n_prefix:-n_tail])   # raw_response
            data[header[6]] = row[-n_tail]                       # cleaned_text
            data[header[7]] = row[-n_tail + 1]                   # cleaned_rating
            data[header[8]] = row[-n_tail + 2]                   # temperature
            data['attempt_type'] = (
                ATTEMPT_CODE_MAP.get(row[-1], 'zero_shot') if has_attempt
                else 'zero_shot'
            )
            recovered.append(data)

    n_recovered = len(recovered)
    if recovered:
        df = pd.concat([df, pd.DataFrame(recovered)], ignore_index=True)

    n_lost = n_file_lines - n_pass1 - n_recovered
    print(f"    Recovered {n_pass1:,} standard + {n_recovered:,} reconstructed"
          f" ({n_lost:,} unrecoverable from multi-line fields) = {len(df):,} total")
    return df


def load_norm_scales():
    """Load validation scales."""
    scales = {}
    if not os.path.exists(SCALE_FILE):
        print(f"Warning: {SCALE_FILE} not found.")
        return scales
    
    try:
        df = pd.read_csv(SCALE_FILE)
        for _, row in df.iterrows():
            norm = row['norm']
            rule = {
                'min': float(row['min_value']) if pd.notna(row['min_value']) else None,
                'max': float(row['max_value']) if pd.notna(row['max_value']) else None,
                'type': row['scale_type'],
                'valid_values': None
            }
            if rule['type'] == 'discrete' and pd.notna(row['valid_values']):
                try:
                    vals = {float(v.strip()) for v in str(row['valid_values']).split(',')}
                    rule['valid_values'] = vals
                except:
                    pass
            scales[norm] = rule
    except Exception as e:
        print(f"Error loading scales: {e}")
    return scales

def get_file_groups(directory):
    """
    Scans directory and groups files by (Model, Condition).
    Condition: 'stochastic' (hitemp/temp>0) or 'deterministic' (std/temp=0).
    Handles multiple file formats: sharded, batched, and offline.
    """
    groups = defaultdict(list)
    files = glob.glob(os.path.join(directory, "*.csv"))
    
    print(f"Scanning {len(files)} files in {directory}...")
    
    for f in files:
        fname = os.path.basename(f)
        matched = False
        
        # Try Pattern 1: Sharded files
        match = SHARD_PATTERN.match(fname)
        if match:
            raw_model = match.group(1).strip('_')
            is_hitemp = bool(match.group(2))
            condition = "stochastic" if is_hitemp else "deterministic"
            groups[(raw_model, condition)].append(f)
            matched = True
        
        # Try Pattern 2: Non-sharded batched files
        if not matched:
            match = BATCHED_PATTERN.match(fname)
            if match:
                raw_model = match.group(1).strip('_')
                is_hitemp = bool(match.group(2))
                condition = "stochastic" if is_hitemp else "deterministic"
                groups[(raw_model, condition)].append(f)
                matched = True
        
        # Try Pattern 3: DAIS offline files
        if not matched:
            match = OFFLINE_PATTERN.match(fname)
            if match:
                raw_model = match.group(1).strip('_')
                temp = float(match.group(2))
                condition = "stochastic" if temp > 0 else "deterministic"
                groups[(raw_model, condition)].append(f)
                matched = True
        
        # Skip test files, manifests, and already-merged files
        if not matched:
            if not any(x in fname for x in ["test_", "manifest", "merged", "missing"]):
                print(f"Skipping unmatched file: {fname}")
    
    return groups


def count_numbers_in_response(text):
    """Counts how many distinct numbers are in the raw response."""
    if not isinstance(text, str):
        return 0
    # Find all integer or decimal numbers
    # Avoid counting inside words, but usually LLM output is separated.
    # Simple regex for numbers
    nums = re.findall(r'-?\d*\.?\d+', text)
    # Filter out common formatting artifacts if strictness is needed, but broad is safer for detection
    # We only care if > 1 to detect verbose ranting that includes multiple ratings
    return len(nums)

# Regex for number extraction
NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")

def smart_parse_rating(raw_response: str, original_rating: str) -> str:
    """
    Re-parse rating from raw_response using smarter logic for reasoning models.
    
    Order of extraction:
    1. Look for 'assistantfinal{N}' or 'final{N}' pattern (GPT-OSS reasoning models)
    2. Look for explicit patterns like 'rating: N', 'I rate N', 'answer: N'
    3. Look for number on the very last line (common in GPT-OSS verbose outputs)
    4. If multiple numbers, take the LAST one
    5. If single number, use it
    6. Fall back to original rating or NO_NUMBER_FOUND
    """
    if not isinstance(raw_response, str):
        return original_rating
    
    raw = raw_response.strip()
    
    # Try 1: Look for "assistantfinal{N}" or "final{N}" pattern (reasoning models)
    final_pattern = re.search(r'(?:assistant)?final\s*(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if final_pattern:
        return final_pattern.group(1)
    
    # Try 2: Look for patterns like "rating: 3" or "I rate 4" or "answer: 5"
    explicit_pattern = re.search(r'(?:rating|rate|answer|respond|say)[:\s]+(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if explicit_pattern:
        return explicit_pattern.group(1)
        
    # Try 3: Look for number on the very last line (common in GPT-OSS)
    # Matches: newline + optional whitespace + number + optional whitespace + end of string
    last_line_pattern = re.search(r'\n\s*(\d+(?:\.\d+)?)\s*$', raw)
    if last_line_pattern:
        return last_line_pattern.group(1)
    
    # Try 4: Find all numbers - if multiple, take the LAST one (likely the answer)
    all_nums = NUM_RE.findall(raw)
    if len(all_nums) > 1:
        return all_nums[-1]  # Last number is usually the final answer
    elif len(all_nums) == 1:
        return all_nums[0]
    
    # Fall back to original rating if we can't extract
    return original_rating if original_rating else "NO_NUMBER_FOUND"


def analyze_and_clean_group(model, condition, file_paths, scales, vocab_list):
    """
    Process one group: Merge -> Clean -> Metric Calc -> Deduplicate -> Save.
    """
    vocab_size = len(vocab_list)
    print(f"\n--- Processing {model} [{condition}] ({len(file_paths)} files) ---")
    
    dfs = []
    for f in file_paths:
        try:
            df = read_raw_csv_robust(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return None

    full_df = pd.concat(dfs, ignore_index=True)
    total_raw_rows = len(full_df)
    
    # 1. Basic Column Checks
    if 'norm' not in full_df.columns or 'word' not in full_df.columns:
        print("Missing 'norm' or 'word' columns.")
        return None

    # 2. Quality & Error Analysis
    
    # Verbose Check (Raw response has multiple numbers?)
    if 'raw_response' in full_df.columns:
        # We assume raw_response extraction already happened but let's check content
        # Actually, the user asked "how many raw responses contain more than one number".
        # This implies we scan 'raw_response' column.
        full_df['num_count'] = full_df['raw_response'].astype(str).apply(count_numbers_in_response)
        full_df['is_verbose'] = full_df['num_count'] > 1
    else:
        full_df['is_verbose'] = False

    # Validation: Parse Errors
    # We expect 'cleaned_rating' to exist from the generation script, or 'rating'?
    # Usually the generation script outputs 'cleaned_rating' alongside 'raw_response'.
    if 'cleaned_rating' not in full_df.columns:
        # Try to use rating or create valid mask
        full_df['cleaned_rating'] = np.nan
        
    full_df['cleaned_rating'] = full_df['cleaned_rating'].astype(str)
    
    # Smart re-parse: Apply improved parsing logic to extract correct rating from verbose responses
    # This handles reasoning models (like GPT-OSS) that output "analysisWe...assistantfinal3"
    # Only apply to GPT-OSS models which use reasoning chains
    if 'raw_response' in full_df.columns and 'gptoss' in model.lower():
        full_df['cleaned_rating'] = full_df.apply(
            lambda row: smart_parse_rating(str(row.get('raw_response', '')), str(row.get('cleaned_rating', ''))),
            axis=1
        ).astype(str)
    
    # Handle Qwen models: strip <think>...</think> blocks and re-parse if needed
    # This mirrors the adapter's strip_think_tags behavior
    if 'raw_response' in full_df.columns and 'qwen' in model.lower():
        THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
        
        def qwen_parse(row):
            raw = str(row.get('raw_response', ''))
            original_rating = str(row.get('cleaned_rating', ''))
            
            # Strip think tags
            cleaned_raw = THINK_RE.sub('', raw).strip()
            
            # If the stripped output is just a number, use it
            if cleaned_raw and cleaned_raw.replace('.', '').replace('-', '').isdigit():
                return cleaned_raw
            
            # Otherwise, use smart_parse_rating on the cleaned response
            return smart_parse_rating(cleaned_raw, original_rating)
        
        full_df['cleaned_rating'] = full_df.apply(qwen_parse, axis=1).astype(str)
        
    # Ensure strict string typing for .str accessor compatibility
    full_df['cleaned_rating'] = full_df['cleaned_rating'].astype(str)
    
    # Define error codes
    error_mask = (
        (full_df['cleaned_rating'] == 'NO_NUMBER_FOUND') |
        (full_df['cleaned_rating'] == 'PARSE_ERROR') | 
        (full_df['cleaned_rating'].str.contains('EXCEPTION', na=False)) |
        (full_df['cleaned_rating'] == 'nan') |
        (full_df['cleaned_rating'] == '')
    )
    
    full_df['rating_val'] = pd.to_numeric(full_df['cleaned_rating'], errors='coerce')
    valid_numeric_mask = full_df['rating_val'].notna()

    # Combine bipolar valence: valence_mohammad = positive - negative
    # Both poles were collected as 0-3 unipolar ratings; their difference yields
    # a -3…+3 bipolar valence norm that maps to valence_mohammad in psychNorms.csv.
    pos_mask = (full_df['norm'] == 'valence_mohammad_positive') & valid_numeric_mask
    neg_mask = (full_df['norm'] == 'valence_mohammad_negative') & valid_numeric_mask

    if pos_mask.any() and neg_mask.any():
        pos_df = full_df[pos_mask].copy()
        neg_df = full_df[neg_mask].copy()
        pos_df['__rep'] = pos_df.groupby('word').cumcount()
        neg_df['__rep'] = neg_df.groupby('word').cumcount()
        merged = pos_df[['word', '__rep', 'rating_val']].merge(
            neg_df[['word', '__rep', 'rating_val']],
            on=['word', '__rep'], suffixes=('_pos', '_neg')
        )
        merged['bipolar_val'] = merged['rating_val_pos'] - merged['rating_val_neg']
        bipolar_df = pos_df.merge(
            merged[['word', '__rep', 'bipolar_val']], on=['word', '__rep']
        ).drop(columns=['__rep'])
        bipolar_df['norm'] = 'valence_mohammad'
        bipolar_df['rating_val'] = bipolar_df['bipolar_val']
        bipolar_df['cleaned_rating'] = bipolar_df['bipolar_val'].astype(str)
        bipolar_df = bipolar_df.drop(columns=['bipolar_val'])
        # Remove ALL rows with positive/negative pole norms (including parse errors)
        all_pos_mask = (full_df['norm'] == 'valence_mohammad_positive')
        all_neg_mask = (full_df['norm'] == 'valence_mohammad_negative')
        full_df = pd.concat(
            [full_df[~all_pos_mask & ~all_neg_mask], bipolar_df], ignore_index=True
        )
        valid_numeric_mask = full_df['rating_val'].notna()
        error_mask = (
            (full_df['cleaned_rating'] == 'NO_NUMBER_FOUND') |
            (full_df['cleaned_rating'] == 'PARSE_ERROR') |
            (full_df['cleaned_rating'].str.contains('EXCEPTION', na=False)) |
            (full_df['cleaned_rating'] == 'nan') |
            (full_df['cleaned_rating'] == '')
        )
        print(f"  Combined valence: {len(bipolar_df)} bipolar rows → norm='valence_mohammad'")
    else:
        if pos_mask.any() or neg_mask.any():
            print("  Warning: only one valence_mohammad pole present; skipping combination.")

    # Outlier Check
    full_df['is_outlier'] = False
    
    def check_outlier(row):
        norm = row['norm']
        val = row['rating_val']
        if pd.isna(val) or norm not in scales:
            return False
        
        rule = scales[norm]
        if rule['min'] is not None and val < rule['min']: return True
        if rule['max'] is not None and val > rule['max']: return True
        if rule['type'] == 'discrete' and rule['valid_values']:
            if val not in rule['valid_values']: return True
        return False

    if scales:
        full_df['is_outlier'] = full_df.apply(check_outlier, axis=1)

    # Error Distribution Analysis
    # Get unique errors from rows that are NOT valid numeric
    invalid_rows = full_df[~valid_numeric_mask]
    # We want to see what strings are in 'cleaned_rating' for these failures
    # Convert to string to handle mixed types safely
    error_counts = invalid_rows['cleaned_rating'].astype(str).value_counts().head(20).to_dict()

    # 3. Aggregating Pre-Deduplication Stats
    stats = {
        'model': model,
        'condition': condition,
        'files': len(file_paths),
        'raw_rows': total_raw_rows,
        'verbose_responses': full_df['is_verbose'].sum(),
        'parse_errors': error_mask.sum(),
        'valid_numeric': valid_numeric_mask.sum(),
        'outliers': full_df['is_outlier'].sum(),
        'error_distribution': error_counts
    }

    # 4. Selection Logic
    
    # Define "Garbage Row"
    # Filter out rows where raw_response is effectively empty AND cleaned_rating has no number
    
    def is_garbage(row):
        # Check raw response emptiness
        raw = str(row.get('raw_response', '')).strip()
        is_raw_empty = (raw == '' or raw == 'nan')
        
        # Check cleaned rating validity
        rating = str(row.get('cleaned_rating', ''))
        is_clean_invalid = ('NO_NUMBER' in rating or 'ERROR' in rating or 'nan' == rating or '' == rating)
        
        return is_raw_empty and is_clean_invalid
    
    # We can use mask for speed
    # conditions: raw_response is null/empty AND cleaned_rating is null/error
    
    raw_response_missing = full_df['raw_response'].isna() | (full_df['raw_response'].astype(str).str.strip() == '')
    
    # reusing error_mask from earlier (defined as NO_NUMBER | PARSE_ERROR | EXCEPTION | nan | '')
    # error_mask is defined around lines 147-153
    
    garbage_mask = raw_response_missing & error_mask
    
    # Filter
    final_df = full_df[~garbage_mask].copy()
    
    expected_reps = 1 if condition == 'deterministic' else 5
    
    # 0 = Best (Valid Not Outlier), 1 = Valid Outlier, 2 = Error
    def get_quality_score(row):
        val = row['rating_val']
        if pd.notna(val) and not row['is_outlier']: return 0
        if pd.notna(val) and row['is_outlier']: return 1
        return 2

    # Vectorized approach for speed
    qual_scores = pd.Series(2, index=final_df.index)
    # Re-compute masks on final_df
    final_valid = final_df['rating_val'].notna()
    final_outlier = final_df['is_outlier']
    
    qual_scores[final_valid & final_outlier] = 1
    qual_scores[final_valid & ~final_outlier] = 0
    final_df['__quality_score'] = qual_scores

    # 5. Post-Deduplication Metrics & Completeness Checks
    norm_metrics = {}
    missing_items = [] 
    
    # Deterministic: Valid Rating + Not Outlier + Non-Empty Raw + Only 1 Number in Raw
    # Stochastic: Valid Rating + Not Outlier (At least 5 reps)
    
    # Base "Good" = Valid Numeric AND Not Outlier (Quality Score 0)
    base_valid = (final_df['__quality_score'] == 0)
    
    if condition == 'deterministic':
        final_df['is_effective_valid'] = base_valid
        target_reps = 1
    else:
        final_df['is_effective_valid'] = base_valid
        target_reps = 5

    # --- CAP to target_reps valid rows per (word, norm) ---
    # Keep first N effectively valid rows per (word, norm) in file order
    # File order favors zero-shot/canonical responses over recovery attempts
    valid_subset = final_df[final_df['is_effective_valid']].copy()
    invalid_subset = final_df[~final_df['is_effective_valid']].copy()

    pre_cap_valid = len(valid_subset)
    valid_subset = valid_subset.groupby(['norm', 'word']).head(target_reps)
    post_cap_valid = len(valid_subset)
    dropped_excess = pre_cap_valid - post_cap_valid

    final_df = pd.concat([valid_subset, invalid_subset], ignore_index=True)
    print(f"  Capped to {target_reps} valid reps/word: "
          f"{post_cap_valid:,} valid ({dropped_excess:,} excess dropped) + "
          f"{len(invalid_subset):,} invalid = {len(final_df):,} total")

    # Pre-calculate counts of effectively valid items per (norm, word)
    valid_counts_series = final_df[final_df['is_effective_valid']].groupby(['norm', 'word']).size()

    for norm, group in final_df.groupby('norm'):
        unique_cues = group['word'].nunique()
        total_rows = len(group)
        
        valid_count = group['rating_val'].notna().sum()
        outlier_count = group['is_outlier'].sum()
        error_count = (group['__quality_score'] == 2).sum()
        verbose_count = group['is_verbose'].sum()
        
        # Calculate Completeness for this Norm based on Target Vocab
        # How many words in VOCAB have >= target_reps valid responses?
        
        # Get counts for this norm specifically
        try:
            current_norm_counts = valid_counts_series.loc[norm]
        except KeyError:
            current_norm_counts = pd.Series(dtype=int)
            
        # Count how many words have >= target_reps
        satisfied_cues_count = (current_norm_counts >= target_reps).sum()
        
        completeness_pct = (satisfied_cues_count / vocab_size) if vocab_size else 0
        missing_cues_count = vocab_size - satisfied_cues_count
        
        norm_metrics[norm] = {
            'unique_cues_found': unique_cues,
            'satisfied_cues': satisfied_cues_count,
            'missing_cues': missing_cues_count,
            'completeness_pct': completeness_pct,
            'valid_rows': valid_count,
            'error_rows': error_count,
            'outlier_rows': outlier_count,
            'verbose_rows': verbose_count,
            'mean_rating': group['rating_val'].mean(),
            'std_rating': group['rating_val'].std()
        }
        
    # Full Missing Items Calculation
    # We Iterate over ALL norms in valid set (Keys from Scales + Keys from Data)
    all_norms = set(final_df['norm'].unique())
    if scales:
        all_norms.update(scales.keys())
        
    valid_norms_str = sorted([n for n in all_norms if isinstance(n, str) and n.strip()])
    
    print(f"Calculating missing items across {len(valid_norms_str)} norms x {vocab_size} words...")
    
    for norm in valid_norms_str:
        try:
            norm_series = valid_counts_series.loc[norm]
        except KeyError:
            norm_series = pd.Series(dtype=int)
            
        # We iterate vocab to find what's missing
        for word in vocab_list:
            cur_count = norm_series.get(word, 0)
            needed = target_reps - cur_count
            
            if needed > 0:
                missing_items.append({
                    "word": word,
                    "norm": norm,
                    "current_valid_reps": cur_count,
                    "target_reps": target_reps,
                    "needed_reps": needed
                })

    # 6. Save Cleaned Data
    # Add explicit model column for downstream analysis
    final_df['model'] = model
    
    # --- REDUCE FILE SIZE FOR VERBOSE MODELS ---
    # For GPT-OSS models (120b, 20b), the raw thinking trace is massive (GBs).
    # If we successfully extracted a valid rating, we don't need the full text anymore.
    # We ONLY drop it if 'rating_val' is valid (not NaN), preserving it for failure analysis.
    if 'gptoss' in model.lower() and 'raw_response' in final_df.columns:
        # Create mask: Is GPT-OSS AND has valid rating
        cleanable_mask = final_df['rating_val'].notna()
        count_cleaned = cleanable_mask.sum()
        if count_cleaned > 0:
            print(f"   -> Optimizing file size: Truncating verbose text for {count_cleaned} valid rows...")
            # Set raw_response to a placeholder for valid rows
            final_df.loc[cleanable_mask, 'raw_response'] = "[VERBOSE_CLEANED]"
    
    # Drop helper cols
    output_cols = [c for c in final_df.columns if not c.startswith('__')]
    save_df = final_df[output_cols]
    
    os.makedirs(os.path.join(CLEAN_DATA_DIR, condition), exist_ok=True)
    out_filename = f"{model}_{condition}.csv"
    out_path = os.path.join(CLEAN_DATA_DIR, condition, out_filename)
    
    save_df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path} ({len(save_df)} rows)")
    
    return stats, norm_metrics, missing_items

def main():
    print(f"Starting Psych Norms Postprocessing Pipeline at {datetime.datetime.now()}")
    
    # Load Resources
    scales = load_norm_scales()
    print(f"Loaded {len(scales)} norm scales.")
    
    try:
        vocab = pd.read_csv(VOCAB_FILE)
        vocab_list = vocab['word'].astype(str).tolist()
        vocab_size = len(vocab_list)
        print(f"Loaded vocab size: {vocab_size}")
    except:
        print("Warning: Could not load vocab file. Completeness percentages will be approximate.")
        vocab_list = []
        vocab_size = 0
        
    # Get Groups
    groups = get_file_groups(RAW_DATA_DIR)
    print(f"Found {len(groups)} Model-Condition pairs.")
    
    all_summaries = []

    csv_report_file = REPORT_FILE.replace(".txt", ".csv")
    csv_cols = ['model', 'condition', 'norm', 
                'unique_cues_found', 'satisfied_cues', 'missing_cues', 'completeness_pct',
                'mean_rating', 'std_rating',
                'norm_valid_rows', 'norm_error_rows', 'norm_outlier_rows', 'norm_verbose_rows',
                'input_files', 'total_raw_rows', 'total_verbose_failures', 'total_parse_errors', 'total_outliers']
    
    # Initialize CSV header - overwrite existing to start fresh
    try:
        pd.DataFrame(columns=csv_cols).to_csv(csv_report_file, index=False)
        print(f"Initialized CSV summary file: {csv_report_file}")
    except Exception as e:
        print(f"Error initializing CSV file: {e}")
    
    with open(REPORT_FILE, 'w') as report:
        report.write("PSYCH NORMS POSTPROCESSING REPORT\n")
        report.write(f"Date: {datetime.datetime.now()}\n")
        report.write("==================================================\n\n")
        
        for (model, condition), file_paths in sorted(groups.items()):
            
            # Run Analysis
            res = analyze_and_clean_group(model, condition, file_paths, scales, vocab_list)
            if not res:
                report.write(f"Model: {model} | Condition: {condition} -> FAILED (No Data)\n\n")
                continue
                
            stats, norm_metrics, missing_items = res

            # Collect data for CSV summary
            for norm, m in norm_metrics.items():
                row = {
                    'model': model,
                    'condition': condition,
                    'norm': norm,
                    'input_files': stats.get('files', 0),
                    'total_raw_rows': stats.get('raw_rows', 0),
                    'total_verbose_failures': stats.get('verbose_responses', 0),
                    'total_parse_errors': stats.get('parse_errors', 0),
                    'total_outliers': stats.get('outliers', 0),
                    'unique_cues_found': m['unique_cues_found'],
                    'satisfied_cues': m['satisfied_cues'],
                    'missing_cues': m['missing_cues'],
                    'completeness_pct': m['completeness_pct'],
                    'norm_valid_rows': m['valid_rows'],
                    'norm_error_rows': m['error_rows'],
                    'norm_outlier_rows': m['outlier_rows'],
                    'norm_verbose_rows': m['verbose_rows'],
                    'mean_rating': m['mean_rating'],
                    'std_rating': m['std_rating']
                }
                
                # Incrementally save to CSV
                try:
                    row_df = pd.DataFrame([row])
                    # Ensure alignment with header columns
                    final_cols = [c for c in csv_cols if c in row_df.columns]
                    # Write without header
                    row_df[final_cols].to_csv(csv_report_file, mode='a', header=False, index=False)
                except Exception as e:
                    print(f"Error appending to CSV: {e}")

                all_summaries.append(row)
            
            # Save Missing Items if any
            if missing_items:
                missing_df = pd.DataFrame(missing_items)
                
                missing_out_path = os.path.join(CLEAN_DATA_DIR, condition, f"missing_items_{model}_{condition}.csv")
                missing_df.to_csv(missing_out_path, index=False)
                print(f"   -> Generated MISSING ITEMS file: {missing_out_path} ({len(missing_df)} tasks)")
            else:
                 print(f"   -> No missing items for {model} {condition}! Complete.")
            
            # Write to Report
            report.write(f"Model: {model} | Condition: {condition}\n")
            report.write(f"  Input Files: {stats['files']}\n")
            report.write(f"  Raw Rows: {stats['raw_rows']:,}\n")
            report.write(f"  Raw Verbose Responses (>1 number): {stats['verbose_responses']:,}\n")
            report.write(f"  Raw Parse Errors: {stats['parse_errors']:,}\n")
            report.write(f"  Raw Outliers: {stats['outliers']:,}\n")
            
            # Print Error Distribution
            if 'error_distribution' in stats and stats['error_distribution']:
                report.write("\n  --- Top Unique Errors (Invalid Ratings) ---\n")
                for err_str, count in stats['error_distribution'].items():
                    # Truncate very long error strings just in case
                    display_str = (err_str[:75] + '..') if len(err_str) > 75 else err_str
                    report.write(f"    {display_str:<80} : {count:,}\n")
            
            report.write("\n  --- Norm Statistics (Cleaned Data & Completeness) ---\n")
            # Table Header
            # Columns: Norm (24), Found (6), Sat (6), Miss (6), %Cmp (6), Valid (7), Err (5), Outl (5), Verb (5), Mean (5)
            header = f"  {'Norm':<24} | {'Found':<6} | {'Sat':<6} | {'Miss':<6} | {'%Cmp':<6} | {'Valid':<7} | {'Err':<5} | {'Outl':<5} | {'Verb':<5} | {'Mean':<5}"
            report.write(f"{header}\n")
            report.write(f"  {'-'*len(header)}\n")
            
            for norm, m in sorted(norm_metrics.items()):
                # Format Percentage
                pct_str = f"{m['completeness_pct']:.1%}"
                report.write(f"  {norm:<24} | {m['unique_cues_found']:<6} | {m['satisfied_cues']:<6} | "
                             f"{m['missing_cues']:<6} | {pct_str:<6} | "
                             f"{m['valid_rows']:<7} | {m['error_rows']:<5} | {m['outlier_rows']:<5} | "
                             f"{m['verbose_rows']:<5} | {m['mean_rating']:<5.2f}\n")
            
            report.write("\n" + "="*50 + "\n\n")

    # Summary CSV already saved incrementally
    print(f"Summary statistics saved incrementally to {csv_report_file}")

    print(f"\nPipeline Complete. Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
