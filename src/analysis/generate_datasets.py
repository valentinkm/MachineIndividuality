#!/usr/bin/env python3
"""
Generate clean supporting dataset CSVs from raw analysis outputs.

Reads from outputs/results/{LMM_Full,LMM_Simulation,EXTENSION,SPECIFICITY}/
and writes clean, self-explanatory CSVs to outputs/datasets/.

Usage:
    python src/analysis/generate_datasets.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "outputs", "datasets")

LMM_DIR = os.path.join(RESULTS_DIR, "LMM_Full")
SIM_DIR = os.path.join(RESULTS_DIR, "LMM_Simulation")
EXT_DIR = os.path.join(RESULTS_DIR, "EXTENSION")
SPEC_DIR = os.path.join(RESULTS_DIR, "SPECIFICITY")

# ── Canonical display names ─────────────────────────────────────────
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


def generate_variance_per_norm():
    """Generate variance_proportions_per_norm.csv from LMM results."""
    src = pd.read_csv(os.path.join(LMM_DIR, "variance_proportions_per_norm.csv"))
    # Exclude MEAN/MEDIAN rows
    src = src[~src["norm"].isin(["MEAN", "MEDIAN"])].copy()
    src = src.sort_values("norm")

    out = pd.DataFrame({
        "norm_id": src["norm"],
        "norm": src["norm"].map(DISPLAY_NORMS),
        "trait_pct": src["prop_trait"],
        "bias_pct": src["prop_bias"],
        "idiosyncrasy_pct": src["prop_idiosyncrasy"],
        "residual_pct": src["prop_residual"],
    })
    out.to_csv(os.path.join(DATASETS_DIR, "variance_proportions_per_norm.csv"), index=False)
    print(f"  ✅ variance_proportions_per_norm.csv ({len(out)} rows)")


def generate_variance_by_dimension():
    """Generate variance_proportions_by_dimension.csv from aggregated LMM results."""
    src = pd.read_csv(os.path.join(LMM_DIR, "variance_proportions_aggregated.csv"))
    # Exclude MEAN/MEDIAN rows
    src = src[~src["dimension"].isin(["MEAN", "MEDIAN"])].copy()
    src = src.sort_values("dimension")

    out = pd.DataFrame({
        "dimension": src["dimension"],
        "trait_pct": src["prop_trait"],
        "bias_pct": src["prop_bias"],
        "idiosyncrasy_pct": src["prop_idiosyncrasy"],
        "residual_pct": src["prop_residual"],
        "n_norms": src["n_norms"].astype(int),
    })
    out.to_csv(os.path.join(DATASETS_DIR, "variance_proportions_by_dimension.csv"), index=False)
    print(f"  ✅ variance_proportions_by_dimension.csv ({len(out)} rows)")


def generate_null_simulation():
    """Generate null_simulation_results.csv from simulation p-values."""
    src = pd.read_csv(os.path.join(SIM_DIR, "simulation_p_values.csv"))
    src = src.sort_values("norm")

    out = pd.DataFrame({
        "norm_id": src["norm"],
        "norm": src["norm"].map(DISPLAY_NORMS),
        "n_simulations": src["n_sim"],
        "observed_interaction_var": src["real_var_interaction"],
        "null_mean_var": src["null_mean_var"],
        "null_max_var": src["null_max_var"],
        "p_value": src["p_value"],
        "z_score": src["z_score"],
    })
    out.to_csv(os.path.join(DATASETS_DIR, "null_simulation_results.csv"), index=False)
    print(f"  ✅ null_simulation_results.csv ({len(out)} rows)")


def generate_specificity_ratios():
    """Generate specificity_ratios.csv by merging idiosyncrasy + ratings specificity."""
    idio = pd.read_csv(os.path.join(SPEC_DIR, "idiosyncrasy_specificity.csv"))
    idio.columns = idio.columns.str.strip()
    idio["prediction_type"] = "idiosyncrasy_blups"

    rat = pd.read_csv(os.path.join(SPEC_DIR, "ratings_specificity.csv"))
    rat.columns = rat.columns.str.strip()
    rat["prediction_type"] = "raw_ratings"

    combined = pd.concat([idio, rat], ignore_index=True)
    combined = combined.sort_values(["prediction_type", "target_model", "target_norm"])

    out = pd.DataFrame({
        "prediction_type": combined["prediction_type"],
        "model_id": combined["target_model"],
        "model": combined["target_model"].map(DISPLAY_MODELS),
        "norm_id": combined["target_norm"],
        "norm": combined["target_norm"].map(DISPLAY_NORMS),
        "within_model_r2": combined["within_r2"],
        "cross_model_mean_r2": combined["cross_mean_r2"],
        "aggregate_r2": combined["aggregate_r2"],
        "specificity_ratio_pairwise": combined["spec_vs_pairwise"],
        "specificity_ratio_aggregate": combined["spec_vs_aggregate"],
    })
    out.to_csv(os.path.join(DATASETS_DIR, "specificity_ratios.csv"), index=False)
    print(f"  ✅ specificity_ratios.csv ({len(out)} rows)")


def generate_human_alignment():
    """Generate merged human_model_alignment.csv from correlation + mode results."""
    # Human correlation results (stochastic mean vs human)
    corr = pd.read_csv(os.path.join(EXT_DIR, "human_correlation_results.csv"))
    # Mode/alignment results (mode, deterministic, KDE peak vs human)
    mode = pd.read_csv(os.path.join(EXT_DIR, "mode_human_alignment.csv"))

    # Merge on (model, norm)
    merged = corr.merge(mode, on=["model", "norm"], how="outer", suffixes=("_corr", "_mode"))

    # Use stochastic mean r from corr (which is the canonical source)
    # and mode/deterministic/kde from mode
    out = pd.DataFrame({
        "model_id": merged["model"],
        "model": merged["model"].map(DISPLAY_MODELS),
        "norm_id": merged["norm"],
        "norm": merged["norm"].map(DISPLAY_NORMS),
        "stochastic_mean_r": merged["correlation"],
        "n_words": merged["n_words_corr"].fillna(merged["n_words_mode"]).astype(int),
        "p_value": merged["p_value"],
        "mode_r": merged["mode_human_corr"],
        "deterministic_r": merged["temp0_human_corr"],
        "kde_peak_r": merged["kde_human_corr"],
        "stochastic_advantage": merged["correlation"] - merged["temp0_human_corr"],
    })
    out = out.sort_values(["model_id", "norm_id"])
    out.to_csv(os.path.join(DATASETS_DIR, "human_model_alignment.csv"), index=False)
    print(f"  ✅ human_model_alignment.csv ({len(out)} rows)")

    # Fisher-z ranking
    def fisher_z(r):
        return np.arctanh(np.clip(r, -0.999, 0.999))

    ranking = []
    for model_id in out["model_id"].unique():
        model_data = out[out["model_id"] == model_id]
        z_values = model_data["stochastic_mean_r"].apply(fisher_z)
        avg_r = np.tanh(z_values.mean())
        ranking.append({
            "model_id": model_id,
            "model": DISPLAY_MODELS.get(model_id, model_id),
            "fisher_z_mean_r": avg_r,
            "n_norms": len(model_data),
            "total_words": int(model_data["n_words"].sum()),
        })

    ranking_df = pd.DataFrame(ranking).sort_values("fisher_z_mean_r", ascending=False)
    ranking_df.to_csv(os.path.join(DATASETS_DIR, "human_alignment_ranking.csv"), index=False)
    print(f"  ✅ human_alignment_ranking.csv ({len(ranking_df)} rows)")


def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    print("=" * 60)
    print("GENERATING SUPPORTING DATASETS")
    print("=" * 60)

    print(f"\nSource: {RESULTS_DIR}")
    print(f"Output: {DATASETS_DIR}\n")

    generate_variance_per_norm()
    generate_variance_by_dimension()
    generate_null_simulation()
    generate_specificity_ratios()
    generate_human_alignment()

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
