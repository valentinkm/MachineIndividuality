#!/usr/bin/env python3
"""
00_prepare_simulation_data.py — Simulation Data Preparation (Pipeline Step 5a)

Reads per-model stochastic clean CSVs, filters to valid observations,
and writes per-norm simulation-ready CSVs containing only the columns
needed by the parametric null simulation (01_parametric_null_simulation.R):
  model, norm, word, rating_val

Inputs:  outputs/raw_behavior/model_norms_clean/stochastic/*_stochastic.csv
Outputs: outputs/results/LMM_Simulation/data_clean/{norm}_sim_ready.csv
"""

import os
import csv
import sys
import glob

# Configuration
INPUT_DIR = "outputs/raw_behavior/model_norms_clean/stochastic"
OUTPUT_DIR = "outputs/results/LMM_Simulation/data_clean"

def main():
    # Increase field size limit for massive 'raw_response' content
    csv.field_size_limit(sys.maxsize)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Clear old files
    print("Clearing old clean files...")
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*_sim_ready.csv")):
        os.remove(f)

    # Get files
    files = glob.glob(os.path.join(INPUT_DIR, "*_stochastic.csv"))
    # Exclude missing items
    files = [f for f in files if "missing_items" not in os.path.basename(f)]
    
    print(f"Found {len(files)} files to process.")
    
    # Note: 'model_key' usually maps to 'model' in R script, but raw file has 'model_key' AND 'model'.
    # We will use 'model' column if present, or 'model_key'.
    # R script used: c("model_key", "norm", "word", "rating_val", "is_effective_valid")
    # Python DictReader keys matches header.
    # Strategy: Dict of {norm_name: file_handle}
    
    norm_handles = {}
    norm_writers = {}
    
    # Header for output
    out_headers = ["model", "norm", "word", "rating_val"]
    
    try:
        for fpath in files:
            fname = os.path.basename(fpath)
            print(f"Processing: {fname} ...")
            
            row_count = 0
            valid_count = 0
            
            try:
                with open(fpath, mode='r', encoding='utf-8', errors='replace') as csvfile:
                    reader = csv.DictReader(csvfile)
                    
                    for row in reader:
                        row_count += 1
                        
                        # Filter Valid
                        # is_effective_valid is string "True"/"False" in CSV? Or 1/0?
                        # In step 1577 head: "True"
                        is_valid = row.get("is_effective_valid", "False")
                        if is_valid not in ["True", "true", "TRUE", "1"]:
                            continue
                            
                        # Extract data
                        norm = row.get("norm", "").strip()
                        if not norm:
                            continue
                            
                        # Get writer for this norm
                        if norm not in norm_handles:
                            p = os.path.join(OUTPUT_DIR, f"{norm}_sim_ready.csv")
                            exists = os.path.exists(p) # Should be false as we cleared
                            fh = open(p, "a", encoding="utf-8", newline="")
                            norm_handles[norm] = fh
                            writer = csv.DictWriter(fh, fieldnames=out_headers)
                            writer.writeheader()
                            norm_writers[norm] = writer
                        
                        # Map row
                        # Prefer 'model' col, fallback 'model_key'
                        model = row.get("model", row.get("model_key", ""))
                        
                        out_row = {
                            "model": model,
                            "norm": norm,
                            "word": row.get("word", ""),
                            "rating_val": row.get("rating_val", "")
                        }
                        
                        norm_writers[norm].writerow(out_row)
                        valid_count += 1
                        
            except Exception as e:
                print(f"   ❌ Error processing file {fname}: {e}")
                continue
                
            print(f"   ✅ Done. Rows: {row_count}. Valid Saved: {valid_count}.")
            
    finally:
        # Close all handles
        print("\nClosing file handles...")
        for norm, fh in norm_handles.items():
            fh.close()
            
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
