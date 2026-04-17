#!/usr/bin/env python3
"""
Shared utilities for analysis scripts.

Provides data loading functions for model norms (stochastic/deterministic),
human norms, norm scales, and LMM results. Used by EXTENSION/, SPECIFICITY/,
and other analysis modules.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Optional, Tuple

# === PATHS ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
MODEL_NORMS_CLEAN = os.path.join(OUTPUTS_DIR, "raw_behavior", "model_norms_clean")
EXTENSION_RESULTS = os.path.join(PROJECT_ROOT, "outputs", "results", "EXTENSION")
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")

# Human norms file
HUMAN_NORMS_FILE = os.path.join(DATA_DIR, "psychNorms.csv")

# LMM results directory (configurable via LMM_OUTPUT_DIR env var)
LMM_FULL_DIR = os.environ.get("LMM_OUTPUT_DIR", os.path.join(RESULTS_DIR, "LMM_Full"))



# Norm scales file
NORM_SCALES_FILE = os.path.join(RESOURCES_DIR, "norm_scales.csv")

# === NORM MAPPING ===
# Maps our internal norm names to human norms CSV column names
# Format: internal_name -> (human_column, source_name)
# Based on actual columns in psychNorms.csv (lowercase with underscores)
NORM_TO_HUMAN_COLUMN = {
    "arousal_warriner": ("arousal_warriner", "Warriner"),
    "concreteness_brysbaert": ("concreteness_brysbaert", "Brysbaert"),
    "aoa_brysbaert": ("aoa_brysbaert", "Brysbaert"),
    "aoa_kuperman": ("aoa_kuperman", "Kuperman"),
    "visual_lancaster": ("visual_lancaster", "Lancaster"),
    "auditory_lancaster": ("auditory_lancaster", "Lancaster"),
    "gustatory_lancaster": ("gustatory_lancaster", "Lancaster"),
    "olfactory_lancaster": ("olfactory_lancaster", "Lancaster"),
    "haptic_lancaster": ("haptic_lancaster", "Lancaster"),
    "humor_engelthaler": ("humor_engelthaler", "Engelthaler"),
    "gender_association_glasgow": ("gender_association_glasgow", "Glasgow"),
    "socialness_diveica": ("socialness_diveica", "Diveica"),
    "morality_troche": ("morality_troche", "Troche"),
    "valence_mohammad": ("valence_mohammad", "Mohammad"),
}


def ensure_results_dir(target_dir: str = "EXTENSION"):
    """Create results directory if it doesn't exist."""
    target_path = os.path.join(PROJECT_ROOT, "outputs", "results", target_dir)
    os.makedirs(target_path, exist_ok=True)
    return target_path


def load_human_norms(sample_n: Optional[int] = None) -> pd.DataFrame:
    """
    Load human psychometric norms from psychNorms.csv.
    
    Args:
        sample_n: If provided, randomly sample this many words for testing.
        
    Returns:
        DataFrame with 'Word' as index and norm columns.
    """
    df = pd.read_csv(HUMAN_NORMS_FILE, low_memory=False)
    
    # The Word column should be the key
    if 'Word' in df.columns:
        df = df.set_index('Word')
    elif 'word' in df.columns:
        df = df.set_index('word')
    
    if sample_n:
        available = min(sample_n, len(df))
        df = df.sample(n=available, random_state=42)
    
    return df


def load_model_norms_stochastic(models: Optional[List[str]] = None, 
                                 sample_n: Optional[int] = None,
                                 nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load stochastic model norms (high temperature, 5 repetitions).
    
    Args:
        models: List of model names to load. If None, load all.
        sample_n: If provided, limit to sample_n unique words (loads full file first).
        nrows: If provided, read only first nrows from each file (fast, for testing).
        
    Returns:
        DataFrame with columns: model, norm, word, rating_val, etc.
    """
    stoch_dir = os.path.join(MODEL_NORMS_CLEAN, "stochastic")
    files = glob.glob(os.path.join(stoch_dir, "*_stochastic.csv"))
    
    # Filter to exclude missing_items files
    files = [f for f in files if "missing_items" not in os.path.basename(f)]
    
    if models:
        files = [f for f in files if any(m in os.path.basename(f) for m in models)]
    
    dfs = []
    for f in files:
        try:
            # Use nrows to limit rows read from disk (much faster than loading all then sampling)
            df = pd.read_csv(f, low_memory=False, nrows=nrows)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure rating_val is numeric
    if 'rating_val' not in full_df.columns and 'cleaned_rating' in full_df.columns:
        full_df['rating_val'] = pd.to_numeric(full_df['cleaned_rating'], errors='coerce')
    
    # Filter out outliers if column exists
    if 'is_outlier' in full_df.columns:
        # Ensure boolean type and handle NaNs to avoid TypeError with ~ operator
        full_df['is_outlier'] = full_df['is_outlier'].fillna(False).astype(bool)
        n_before = len(full_df)
        full_df = full_df[~full_df['is_outlier']]
        print(f"  Outlier filter: {n_before} → {len(full_df)} rows ({n_before - len(full_df)} out-of-scale removed)")
    
    # Filter to effectively valid rows (matches LMM upstream: is_effective_valid == TRUE)
    if 'is_effective_valid' in full_df.columns:
        full_df['is_effective_valid'] = full_df['is_effective_valid'].fillna(False).astype(bool)
        n_before = len(full_df)
        full_df = full_df[full_df['is_effective_valid']]
        print(f"  Validity filter: {n_before} → {len(full_df)} rows ({n_before - len(full_df)} invalid removed)")
    
    if sample_n and 'word' in full_df.columns:
        unique_words = full_df['word'].unique()
        if len(unique_words) > sample_n:
            sampled_words = np.random.choice(unique_words, size=sample_n, replace=False)
            full_df = full_df[full_df['word'].isin(sampled_words)]
    
    return full_df


def load_model_norms_deterministic(models: Optional[List[str]] = None,
                                    sample_n: Optional[int] = None,
                                    nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load deterministic model norms (temperature 0).
    
    Args:
        models: List of model names to load. If None, load all.
        sample_n: If provided, limit to sample_n unique words.
        nrows: If provided, read only first nrows from each file (fast, for testing).
        
    Returns:
        DataFrame with columns: model, norm, word, rating_val, etc.
    """
    det_dir = os.path.join(MODEL_NORMS_CLEAN, "deterministic")
    files = glob.glob(os.path.join(det_dir, "*_deterministic.csv"))
    
    files = [f for f in files if "missing_items" not in os.path.basename(f)]
    
    if models:
        files = [f for f in files if any(m in os.path.basename(f) for m in models)]
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, nrows=nrows)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    if 'rating_val' not in full_df.columns and 'cleaned_rating' in full_df.columns:
        full_df['rating_val'] = pd.to_numeric(full_df['cleaned_rating'], errors='coerce')
    
    # Filter out outliers if column exists
    if 'is_outlier' in full_df.columns:
        # Ensure boolean type and handle NaNs to avoid TypeError with ~ operator
        full_df['is_outlier'] = full_df['is_outlier'].fillna(False).astype(bool)
        full_df = full_df[~full_df['is_outlier']]
    
    if sample_n and 'word' in full_df.columns:
        unique_words = full_df['word'].unique()
        if len(unique_words) > sample_n:
            sampled_words = np.random.choice(unique_words, size=sample_n, replace=False)
            full_df = full_df[full_df['word'].isin(sampled_words)]
    
    return full_df


def compute_model_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean rating per (model, norm, word) from stochastic data.
    
    Returns:
        DataFrame with columns: model, norm, word, mean_rating
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to valid ratings only
    valid_df = df[df['rating_val'].notna()].copy()
    
    means = valid_df.groupby(['model', 'norm', 'word'])['rating_val'].mean().reset_index()
    means = means.rename(columns={'rating_val': 'mean_rating'})
    
    return means


def load_norm_scales() -> Dict[str, Dict]:
    """
    Load norm scale definitions (min, max, type, valid_values).
    
    Returns:
        Dict mapping norm name to scale info.
    """
    scales = {}
    if not os.path.exists(NORM_SCALES_FILE):
        print(f"Warning: {NORM_SCALES_FILE} not found")
        return scales
    
    try:
        df = pd.read_csv(NORM_SCALES_FILE)
        for _, row in df.iterrows():
            norm = row['norm']
            scales[norm] = {
                'min': float(row['min_value']) if pd.notna(row.get('min_value')) else None,
                'max': float(row['max_value']) if pd.notna(row.get('max_value')) else None,
                'type': row.get('scale_type', 'continuous'),
                'valid_values': None
            }
            if scales[norm]['type'] == 'discrete' and pd.notna(row.get('valid_values')):
                try:
                    vals = {float(v.strip()) for v in str(row['valid_values']).split(',')}
                    scales[norm]['valid_values'] = vals
                except:
                    pass
    except Exception as e:
        print(f"Error loading scales: {e}")
    
    return scales


def get_available_models() -> List[str]:
    """Get list of models with available stochastic data."""
    stoch_dir = os.path.join(MODEL_NORMS_CLEAN, "stochastic")
    files = glob.glob(os.path.join(stoch_dir, "*_stochastic.csv"))
    files = [f for f in files if "missing_items" not in os.path.basename(f)]
    
    models = []
    for f in files:
        basename = os.path.basename(f)
        model = basename.replace("_stochastic.csv", "")
        models.append(model)
    
    return sorted(models)


def get_available_norms(df: pd.DataFrame) -> List[str]:
    """Get list of unique norms in a dataframe."""
    if 'norm' in df.columns:
        return sorted(df['norm'].unique().tolist())
    return []


if __name__ == "__main__":
    # Quick test
    print("Testing utils...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Available stochastic models: {get_available_models()}")
    
    print("\nLoading sample stochastic data (100 words)...")
    df = load_model_norms_stochastic(sample_n=100)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Norms: {get_available_norms(df)}")
