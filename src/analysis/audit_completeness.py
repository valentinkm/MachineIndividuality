#!/usr/bin/env python3
"""
Data Completeness Audit

Audits the postprocessed model norms data at two layers:
  Layer 1 — Dictionary coverage: For each (model, norm, condition), what
            fraction of the 107,083 target vocabulary words received at
            least one valid response?
  Layer 2 — Repetition completeness (stochastic only): For each
            (model, norm), what fraction of words achieved exactly 5
            valid repetitions? Also reports the full distribution of
            repetition counts (0–5+).

Outputs (→ outputs/datasets/):
  - completeness_dictionary.csv     Layer 1: per (model, norm, condition)
  - completeness_repetitions.csv    Layer 2: per (model, norm) rep distribution

Usage:
    python src/analysis/audit_completeness.py
"""

import pandas as pd
import numpy as np
import os
import glob

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_DIR = os.path.join(PROJECT_ROOT, "outputs", "raw_behavior", "model_norms_clean")
STOCH_DIR = os.path.join(CLEAN_DIR, "stochastic")
DET_DIR = os.path.join(CLEAN_DIR, "deterministic")
VOCAB_FILE = os.path.join(PROJECT_ROOT, "resources", "psychNorms_vocab.csv")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "outputs", "datasets")

TARGET_REPS = 5  # target repetitions for stochastic condition

# ── Display names ───────────────────────────────────────────────────────
DISPLAY_MODELS = {
    "qwen32b": "Qwen3-32B",
    "qwen3_235b_it": "Qwen3-235B-A22B",
    "mistral24b": "Mistral-Small-24B",
    "gemma27b": "gemma-3-27b-it",
    "gptoss_20b": "gpt-oss-20b",
    "gptoss_120b": "gpt-oss-120b",
    "olmo3_32b_it": "OLMo-3.1-32B",
    "falcon_h1_34b_it": "Falcon-H1-34B",
    "granite_4_small": "granite-4.0-h-small",
    "phi_4": "phi-4",
}

DISPLAY_NORMS = {
    "arousal_warriner": "Arousal (Warriner)",
    "concreteness_brysbaert": "Concreteness (Brysbaert)",
    "valence_mohammad": "Valence (Mohammad)",
    "visual_lancaster": "Visual (Lancaster)",
    "auditory_lancaster": "Auditory (Lancaster)",
    "gustatory_lancaster": "Gustatory (Lancaster)",
    "olfactory_lancaster": "Olfactory (Lancaster)",
    "haptic_lancaster": "Haptic (Lancaster)",
    "aoa_kuperman": "AoA (Kuperman)",
    "aoa_brysbaert": "AoA (Brysbaert)",
    "morality_troche": "Morality (Troché)",
    "gender_association_glasgow": "Gender Association (Glasgow)",
    "humor_engelthaler": "Humor (Engelthaler)",
    "socialness_diveica": "Socialness (Diveica)",
}


def load_vocab() -> set:
    """Load the target vocabulary (107,083 words)."""
    vocab = pd.read_csv(VOCAB_FILE)
    words = set(vocab["word"].astype(str).tolist())
    print(f"Target vocabulary: {len(words):,} words")
    return words


def load_clean_files(data_dir: str) -> pd.DataFrame:
    """Load all clean model CSVs from a directory (stochastic or deterministic)."""
    pattern = os.path.join(data_dir, "*_*.csv")
    files = sorted([f for f in glob.glob(pattern) if "missing_items" not in f])

    if not files:
        print(f"  WARNING: No files found in {data_dir}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=["model", "norm", "word", "is_effective_valid"],
                         low_memory=False)
        dfs.append(df)
        print(f"  {os.path.basename(f):45s} {len(df):>10,} rows")

    return pd.concat(dfs, ignore_index=True)


def audit_dictionary_coverage(df: pd.DataFrame, vocab: set,
                               condition: str) -> pd.DataFrame:
    """
    Layer 1: Dictionary coverage.
    For each (model, norm), count how many of the target vocab words
    have at least one valid response.
    """
    vocab_size = len(vocab)
    results = []

    # Only valid responses count
    valid_df = df[df["is_effective_valid"]].copy()

    for (model, norm), group in valid_df.groupby(["model", "norm"]):
        words_with_valid = set(group["word"].unique())
        covered = words_with_valid & vocab          # in-vocab words with response
        n_covered = len(covered)
        n_missing = vocab_size - n_covered
        # words responded to but not in vocab (shouldn't happen, but check)
        n_extra = len(words_with_valid - vocab)

        # Total responses (valid + invalid) for this model-norm
        total_all = len(df[(df["model"] == model) & (df["norm"] == norm)])
        total_valid = len(group)

        results.append({
            "model_id": model,
            "model": DISPLAY_MODELS.get(model, model),
            "norm_id": norm,
            "norm": DISPLAY_NORMS.get(norm, norm),
            "condition": condition,
            "vocab_size": vocab_size,
            "n_words_covered": n_covered,
            "n_words_missing": n_missing,
            "coverage_pct": n_covered / vocab_size,
            "n_total_responses": total_all,
            "n_valid_responses": total_valid,
            "exclusion_rate": 1 - (total_valid / total_all) if total_all > 0 else 0,
        })

    return pd.DataFrame(results).sort_values(["condition", "model_id", "norm_id"])


def audit_repetition_completeness(df: pd.DataFrame, vocab: set) -> pd.DataFrame:
    """
    Layer 2: Repetition completeness (stochastic only).
    For each (model, norm), count how many valid reps each vocab word has
    (0 through 5+) and report the distribution.
    """
    vocab_size = len(vocab)
    results = []

    valid_df = df[df["is_effective_valid"]].copy()

    for (model, norm), group in valid_df.groupby(["model", "norm"]):
        # Count valid reps per word
        reps_per_word = group.groupby("word").size()

        # Map all vocab words to their rep count (0 if missing)
        vocab_list = sorted(vocab)
        rep_counts = pd.Series(0, index=vocab_list, dtype=int)
        rep_counts.update(reps_per_word)
        # Cap display at TARGET_REPS for the distribution
        rep_counts_capped = rep_counts.clip(upper=TARGET_REPS)

        # Distribution: how many words have 0, 1, 2, 3, 4, 5 reps
        dist = rep_counts_capped.value_counts().sort_index()
        for r in range(TARGET_REPS + 1):
            if r not in dist.index:
                dist[r] = 0
        dist = dist.sort_index()

        # Also count overcomplete (>5 before capping, shouldn't happen post-cap)
        n_overcomplete = int((reps_per_word > TARGET_REPS).sum())

        n_complete = int(dist.get(TARGET_REPS, 0))
        n_undercomplete = vocab_size - n_complete  # words with < 5 reps

        results.append({
            "model_id": model,
            "model": DISPLAY_MODELS.get(model, model),
            "norm_id": norm,
            "norm": DISPLAY_NORMS.get(norm, norm),
            "vocab_size": vocab_size,
            "n_words_0_reps": int(dist.get(0, 0)),
            "n_words_1_reps": int(dist.get(1, 0)),
            "n_words_2_reps": int(dist.get(2, 0)),
            "n_words_3_reps": int(dist.get(3, 0)),
            "n_words_4_reps": int(dist.get(4, 0)),
            "n_words_5_reps": int(dist.get(5, 0)),
            "n_overcomplete": n_overcomplete,
            "completeness_pct": n_complete / vocab_size,
            "mean_reps": float(rep_counts.mean()),
            "median_reps": float(rep_counts.median()),
        })

    return pd.DataFrame(results).sort_values(["model_id", "norm_id"])


def main():
    np.random.seed(42)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    print("=" * 70)
    print("DATA COMPLETENESS AUDIT")
    print("=" * 70)

    vocab = load_vocab()

    # ── Layer 0+1: Stochastic dictionary coverage ──────────────────────
    print("\n─── Stochastic data ───")
    stoch = load_clean_files(STOCH_DIR)
    stoch_dict = pd.DataFrame()
    if not stoch.empty:
        stoch_dict = audit_dictionary_coverage(stoch, vocab, "stochastic")
        print(f"\n  Layer 1 (dictionary): {len(stoch_dict)} model-norm pairs")

    # ── Layer 0+1: Deterministic dictionary coverage ───────────────────
    print("\n─── Deterministic data ───")
    det = load_clean_files(DET_DIR)
    det_dict = pd.DataFrame()
    if not det.empty:
        det_dict = audit_dictionary_coverage(det, vocab, "deterministic")
        print(f"\n  Layer 1 (dictionary): {len(det_dict)} model-norm pairs")

    # ── Write Layer 1 ──────────────────────────────────────────────────
    dict_combined = pd.concat([stoch_dict, det_dict], ignore_index=True)
    dict_path = os.path.join(DATASETS_DIR, "completeness_dictionary.csv")
    dict_combined.to_csv(dict_path, index=False)
    print(f"\n✅ {dict_path} ({len(dict_combined)} rows)")

    # ── Layer 2: Stochastic repetition completeness ────────────────────
    print("\n─── Layer 2: Repetition completeness (stochastic) ───")
    if not stoch.empty:
        rep_df = audit_repetition_completeness(stoch, vocab)
        rep_path = os.path.join(DATASETS_DIR, "completeness_repetitions.csv")
        rep_df.to_csv(rep_path, index=False)
        print(f"\n✅ {rep_path} ({len(rep_df)} rows)")
    else:
        rep_df = pd.DataFrame()

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not stoch_dict.empty:
        print("\n── Stochastic dictionary coverage ──")
        for model in sorted(stoch_dict["model"].unique()):
            sub = stoch_dict[stoch_dict["model"] == model]
            avg_cov = sub["coverage_pct"].mean()
            min_cov = sub["coverage_pct"].min()
            avg_excl = sub["exclusion_rate"].mean()
            print(f"  {model:30s}  coverage={avg_cov:.4%}  "
                  f"min={min_cov:.4%}  excl={avg_excl:.4%}")

    if not det_dict.empty:
        print("\n── Deterministic dictionary coverage ──")
        for model in sorted(det_dict["model"].unique()):
            sub = det_dict[det_dict["model"] == model]
            avg_cov = sub["coverage_pct"].mean()
            min_cov = sub["coverage_pct"].min()
            avg_excl = sub["exclusion_rate"].mean()
            print(f"  {model:30s}  coverage={avg_cov:.4%}  "
                  f"min={min_cov:.4%}  excl={avg_excl:.4%}")

    if not rep_df.empty:
        print("\n── Stochastic repetition completeness (5 reps target) ──")
        for model in sorted(rep_df["model"].unique()):
            sub = rep_df[rep_df["model"] == model]
            avg_compl = sub["completeness_pct"].mean()
            min_compl = sub["completeness_pct"].min()
            avg_reps = sub["mean_reps"].mean()
            total_0 = sub["n_words_0_reps"].sum()
            print(f"  {model:30s}  5-rep={avg_compl:.4%}  "
                  f"min={min_compl:.4%}  avg_reps={avg_reps:.2f}  "
                  f"words_w_0={total_0:,}")

    print()


if __name__ == "__main__":
    main()
