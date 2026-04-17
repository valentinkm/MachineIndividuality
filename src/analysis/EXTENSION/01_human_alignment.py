#!/usr/bin/env python3
"""
Human Alignment Analysis

Computes per-model × per-norm Pearson r against human norms for both stochastic
mean and deterministic ratings in one pass, produces the Fisher-z aggregation,
and outputs the merged dataset CSVs.

Usage:
    python 01_human_alignment.py [--sample N] [--workers N]
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_human_norms,
    load_model_norms_stochastic,
    load_model_norms_deterministic,
    load_norm_scales,
    compute_model_means,
    get_available_models,
    get_available_norms,
    ensure_results_dir,
    NORM_TO_HUMAN_COLUMN,
    LMM_FULL_DIR,
    MODEL_NORMS_CLEAN,
)

# ═══════════════════════════════════════════════════════════════════════
# Display name mapping (from 06_combined_figure_publication.R)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Norm mapping
# ═══════════════════════════════════════════════════════════════════════

def discover_human_columns(human_df: pd.DataFrame, model_norms: List[str]) -> Dict[str, str]:
    """Auto-discover which human norm columns correspond to model norms."""
    mapping = {}
    available_cols = set(human_df.columns)

    for model_norm in model_norms:
        if model_norm in NORM_TO_HUMAN_COLUMN:
            human_col, _ = NORM_TO_HUMAN_COLUMN[model_norm]
            if human_col in available_cols:
                mapping[model_norm] = human_col
            else:
                print(f"Warning: Expected column '{human_col}' for norm '{model_norm}' not in human data")
        else:
            if model_norm in available_cols:
                mapping[model_norm] = model_norm

    return mapping


# ═══════════════════════════════════════════════════════════════════════
# Stochastic statistics (mode, KDE peak)
# ═══════════════════════════════════════════════════════════════════════

def compute_mode_discrete(values: np.ndarray) -> float:
    """Compute mode for discrete values."""
    if len(values) == 0:
        return np.nan
    counts = Counter(values)
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else np.nan


def compute_kde_peak(values: np.ndarray) -> float:
    """Compute KDE peak for continuous values."""
    if len(values) < 2 or np.std(values) == 0:
        return np.mean(values) if len(values) > 0 else np.nan
    try:
        kde = gaussian_kde(values)
        x_min, x_max = values.min() - 0.5, values.max() + 0.5
        x_eval = np.linspace(x_min, x_max, 100)
        kde_vals = kde(x_eval)
        return x_eval[np.argmax(kde_vals)]
    except Exception:
        return np.mean(values)


def process_stochastic_group(args: Tuple) -> Optional[Dict]:
    """Process a single (model, norm, word) group for stochastic statistics."""
    model, norm, word, values, is_discrete = args

    n_reps = len(values)
    if n_reps == 0:
        return None

    mean_val = np.mean(values)
    std_val = np.std(values) if n_reps > 1 else 0

    if is_discrete:
        mode_val = compute_mode_discrete(values)
        kde_peak = mode_val
    else:
        mode_val = compute_mode_discrete(np.round(values))
        kde_peak = compute_kde_peak(values)

    return {
        'model': model,
        'norm': norm,
        'word': word,
        'mode': mode_val,
        'kde_peak': kde_peak,
        'mean': mean_val,
        'std': std_val,
        'n_reps': n_reps
    }


def compute_stochastic_statistics(stoch_df: pd.DataFrame,
                                   norm_scales: Dict,
                                   max_workers: int = 4) -> pd.DataFrame:
    """Compute mode, KDE peak, and mean for each (model, norm, word)."""
    valid_df = stoch_df[stoch_df['rating_val'].notna()].copy()

    tasks = []
    for (model, norm, word), group in valid_df.groupby(['model', 'norm', 'word']):
        values = group['rating_val'].values
        is_discrete = norm in norm_scales and norm_scales[norm].get('type') == 'discrete'
        tasks.append((model, norm, word, values, is_discrete))

    print(f"   Processing {len(tasks):,} word-level statistics with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.map(process_stochastic_group, tasks, chunksize=1000))
        results = [r for r in futures if r is not None]

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# Unified human alignment computation
# ═══════════════════════════════════════════════════════════════════════

def compute_human_alignment_unified(stoch_stats: pd.DataFrame,
                                     det_df: pd.DataFrame,
                                     human_df: pd.DataFrame,
                                     norm_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Compute human correlation for stochastic mean, mode, KDE peak, AND
    deterministic ratings in a single pass.

    Returns one row per (model, norm) with all correlation metrics.
    """
    # Merge deterministic data
    valid_det = det_df[det_df['rating_val'].notna()].copy()
    valid_det['rating_val'] = pd.to_numeric(valid_det['rating_val'], errors='coerce')
    det_means = valid_det.groupby(['model', 'norm', 'word'])['rating_val'].mean().reset_index()
    det_means = det_means.rename(columns={'rating_val': 'temp0'})
    merged = pd.merge(stoch_stats, det_means, on=['model', 'norm', 'word'], how='left')
    merged['temp0'] = pd.to_numeric(merged['temp0'], errors='coerce')

    # Reverse arousal model ratings to match recoded human norms
    # Model prompt uses original ANEW: 1=excited, 9=calm
    # Human norms in psychNorms.csv were recoded by Warriner et al.: 1=calm, 9=excited
    arousal_mask = merged['norm'] == 'arousal_warriner'
    for col in ['mean', 'mode', 'kde_peak']:
        merged.loc[arousal_mask, col] = 10 - merged.loc[arousal_mask, col]
    if 'temp0' in merged.columns:
        merged.loc[arousal_mask, 'temp0'] = 10 - merged.loc[arousal_mask, 'temp0']

    results = []

    for (model, norm), group in merged.groupby(['model', 'norm']):
        if norm not in norm_mapping:
            continue

        human_col = norm_mapping[norm]
        group = group.set_index('word')
        overlap = set(group.index) & set(human_df.index)

        if len(overlap) < 10:
            continue

        group_overlap = group.loc[list(overlap)]
        human_overlap = human_df.loc[list(overlap), human_col]

        valid = group_overlap['mean'].notna() & human_overlap.notna()
        if valid.sum() < 10:
            continue

        # Compute all correlations in one pass
        stoch_mean_r, stoch_mean_p = stats.pearsonr(
            group_overlap.loc[valid, 'mean'], human_overlap[valid])
        mode_r, _ = stats.pearsonr(
            group_overlap.loc[valid, 'mode'], human_overlap[valid])
        kde_r, _ = stats.pearsonr(
            group_overlap.loc[valid, 'kde_peak'], human_overlap[valid])

        # Deterministic
        valid_det_mask = valid & group_overlap['temp0'].notna()
        if valid_det_mask.sum() >= 10:
            temp0_r, _ = stats.pearsonr(
                group_overlap.loc[valid_det_mask, 'temp0'], human_overlap[valid_det_mask])
        else:
            temp0_r = np.nan

        results.append({
            'model': model,
            'norm': norm,
            'human_source': human_col,
            'correlation': stoch_mean_r,
            'n_words': int(valid.sum()),
            'p_value': stoch_mean_p,
            'mode_human_corr': mode_r,
            'mean_human_corr': stoch_mean_r,
            'temp0_human_corr': temp0_r,
            'kde_human_corr': kde_r,
        })

    return pd.DataFrame(results)


def compute_fisher_z_ranking(results: pd.DataFrame) -> pd.DataFrame:
    """Compute Fisher-z aggregated model ranking."""
    if results.empty:
        return pd.DataFrame()

    def fisher_z(r):
        return np.arctanh(np.clip(r, -0.999, 0.999))

    agg = []
    for model in results['model'].unique():
        model_data = results[results['model'] == model]
        z_values = model_data['correlation'].apply(fisher_z)
        avg_r = np.tanh(z_values.mean())
        agg.append({
            'model': model,
            'avg_correlation': avg_r,
            'n_norms': len(model_data),
            'total_words': model_data['n_words'].sum()
        })

    return pd.DataFrame(agg).sort_values('avg_correlation', ascending=False)


def compute_mode_alignment_metrics(stoch_stats: pd.DataFrame,
                                    det_df: pd.DataFrame) -> pd.DataFrame:
    """Compute alignment metrics between stochastic summaries and deterministic."""
    valid_det = det_df[det_df['rating_val'].notna()].copy()
    valid_det['rating_val'] = pd.to_numeric(valid_det['rating_val'], errors='coerce')
    det_means = valid_det.groupby(['model', 'norm', 'word'])['rating_val'].mean().reset_index()
    det_means = det_means.rename(columns={'rating_val': 'temp0'})
    merged = pd.merge(stoch_stats, det_means, on=['model', 'norm', 'word'], how='left')
    merged['temp0'] = pd.to_numeric(merged['temp0'], errors='coerce')

    results = []
    for (model, norm), group in merged.groupby(['model', 'norm']):
        if len(group) < 10:
            continue

        # Drop rows with NaN temp0 (models without deterministic data)
        valid = group.dropna(subset=['temp0', 'mode', 'mean', 'kde_peak'])
        if len(valid) < 10:
            continue

        mode_temp0_corr, _ = stats.pearsonr(valid['mode'], valid['temp0'])
        mean_temp0_corr, _ = stats.pearsonr(valid['mean'], valid['temp0'])
        kde_temp0_corr, _ = stats.pearsonr(valid['kde_peak'], valid['temp0'])
        mode_temp0_mae = np.abs(valid['mode'] - valid['temp0']).mean()
        mean_temp0_mae = np.abs(valid['mean'] - valid['temp0']).mean()
        kde_temp0_mae = np.abs(valid['kde_peak'] - valid['temp0']).mean()
        mode_agreement_rate = (np.round(valid['mode']) == np.round(valid['temp0'])).mean()

        results.append({
            'model': model, 'norm': norm,
            'mode_temp0_corr': mode_temp0_corr,
            'mean_temp0_corr': mean_temp0_corr,
            'kde_temp0_corr': kde_temp0_corr,
            'mode_temp0_mae': mode_temp0_mae,
            'mean_temp0_mae': mean_temp0_mae,
            'kde_temp0_mae': kde_temp0_mae,
            'mode_agreement_rate': mode_agreement_rate,
            'n_words': len(group),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# Plotting (carried over from original scripts)
# ═══════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(results: pd.DataFrame, output_path: str):
    """Create heatmap of model × norm correlations."""
    pivot = results.pivot(index='model', columns='norm', values='correlation')
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson r'},
                annot_kws={'size': 8})
    ax.set_title('Model-Human Correlation by Norm', fontsize=14)
    ax.set_xlabel('Norm')
    ax.set_ylabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")


def plot_model_ranking(agg_results: pd.DataFrame, output_path: str):
    """Create bar plot of model ranking by average human correlation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(agg_results)))
    ax.barh(agg_results['model'], agg_results['avg_correlation'], color=colors)
    ax.set_xlabel('Average Correlation with Human Norms')
    ax.set_ylabel('Model')
    ax.set_title('Model Ranking by Human Alignment')
    ax.set_xlim(0, 1)
    for i, (_, row) in enumerate(agg_results.iterrows()):
        ax.text(row['avg_correlation'] + 0.01, i, f"{row['avg_correlation']:.3f}",
                va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ranking plot to {output_path}")


def plot_human_alignment_comparison(human_align_df: pd.DataFrame, output_path: str):
    """Compare mode/mean/temp0 correlations with human data."""
    fig, ax = plt.subplots(figsize=(14, 7))
    models = sorted(human_align_df['model'].unique())
    x = np.arange(len(models))
    width = 0.2

    mode_corrs = [human_align_df[human_align_df['model'] == m]['mode_human_corr'].mean() for m in models]
    mean_corrs = [human_align_df[human_align_df['model'] == m]['mean_human_corr'].mean() for m in models]
    temp0_corrs = [human_align_df[human_align_df['model'] == m]['temp0_human_corr'].mean() for m in models]
    kde_corrs = [human_align_df[human_align_df['model'] == m]['kde_human_corr'].mean() for m in models]

    ax.bar(x - 1.5*width, mode_corrs, width, label='Mode-Human', color='steelblue')
    ax.bar(x - 0.5*width, mean_corrs, width, label='Mean-Human', color='darkorange')
    ax.bar(x + 0.5*width, temp0_corrs, width, label='Temp0-Human', color='forestgreen')
    ax.bar(x + 1.5*width, kde_corrs, width, label='KDE Peak-Human', color='purple')

    ax.set_ylabel('Correlation with Human Norms')
    ax.set_xlabel('Model')
    ax.set_title('Which Summary Statistic Best Predicts Human Judgments?')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved human alignment plot to {output_path}")


def plot_stacked_by_norm(results: pd.DataFrame, output_path: str):
    """Create grouped bar chart showing model correlations stacked by norm."""
    fig, ax = plt.subplots(figsize=(18, 10))
    norms = sorted(results['norm'].unique())
    models = sorted(results['model'].unique())
    n_models = len(models)
    x = np.arange(len(norms))
    width = 0.8 / n_models
    cmap = plt.cm.get_cmap('tab10')

    for i, model in enumerate(models):
        model_data = results[results['model'] == model]
        correlations = []
        for norm in norms:
            vals = model_data[model_data['norm'] == norm]['correlation'].values
            correlations.append(vals[0] if len(vals) > 0 else 0)
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, correlations, width, label=model, color=cmap(i % 10))

    ax.set_xlabel('Norm')
    ax.set_ylabel('Correlation with Human Data')
    ax.set_title('Model-Human Correlations by Norm')
    ax.set_xticks(x)
    ax.set_xticklabels(norms, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-0.2, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved stacked plot to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Dataset output (clean CSVs for outputs/datasets/)
# ═══════════════════════════════════════════════════════════════════════

def write_dataset_csvs(results: pd.DataFrame, ranking: pd.DataFrame,
                        datasets_dir: str):
    """Write clean dataset CSVs for transparency/replication."""
    os.makedirs(datasets_dir, exist_ok=True)

    # Merged human alignment
    out = pd.DataFrame({
        "model_id": results["model"],
        "model": results["model"].map(DISPLAY_MODELS),
        "norm_id": results["norm"],
        "norm": results["norm"].map(DISPLAY_NORMS),
        "stochastic_mean_r": results["correlation"],
        "n_words": results["n_words"],
        "p_value": results["p_value"],
        "mode_r": results["mode_human_corr"],
        "deterministic_r": results["temp0_human_corr"],
        "kde_peak_r": results["kde_human_corr"],
        "stochastic_advantage": results["correlation"] - results["temp0_human_corr"],
    })
    out = out.sort_values(["model_id", "norm_id"])
    out.to_csv(os.path.join(datasets_dir, "human_model_alignment.csv"), index=False)
    print(f"Saved dataset: {os.path.join(datasets_dir, 'human_model_alignment.csv')}")

    # Ranking
    rank_out = pd.DataFrame({
        "model_id": ranking["model"],
        "model": ranking["model"].map(DISPLAY_MODELS),
        "fisher_z_mean_r": ranking["avg_correlation"],
        "n_norms": ranking["n_norms"],
        "total_words": ranking["total_words"],
    })
    rank_out.to_csv(os.path.join(datasets_dir, "human_alignment_ranking.csv"), index=False)
    print(f"Saved dataset: {os.path.join(datasets_dir, 'human_alignment_ranking.csv')}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Human Alignment Analysis')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N words for testing (default: use all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--nrows', type=int, default=None,
                        help='Read only first N rows per file (fast testing)')
    args = parser.parse_args()

    results_dir = ensure_results_dir()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    datasets_dir = os.path.join(project_root, "outputs", "datasets")

    print("=" * 60)
    print("HUMAN ALIGNMENT ANALYSIS")
    print("=" * 60)

    # 1. Load norm scales
    print("\n1. Loading norm scales...")
    norm_scales = load_norm_scales()
    print(f"   Loaded {len(norm_scales)} scales")

    # 2. Load stochastic data
    print("\n2. Loading stochastic model norms...")
    stoch_df = load_model_norms_stochastic(sample_n=args.sample, nrows=args.nrows)
    print(f"   Loaded {len(stoch_df):,} rows")

    # 3. Load deterministic data
    print("\n3. Loading deterministic model norms...")
    det_df = load_model_norms_deterministic(sample_n=args.sample, nrows=args.nrows)
    print(f"   Loaded {len(det_df):,} rows")

    # 4. Compute stochastic statistics (mode, KDE peak, mean)
    print("\n4. Computing stochastic statistics (mode/KDE-peak/mean)...")
    stoch_stats = compute_stochastic_statistics(stoch_df, norm_scales, args.workers)
    print(f"   Computed stats for {len(stoch_stats):,} (model, norm, word) combinations")

    # 5. Load human norms and discover mapping
    print("\n5. Loading human norms...")
    human_df = load_human_norms()
    model_norms = get_available_norms(stoch_df)
    norm_mapping = discover_human_columns(human_df, model_norms)
    print(f"   Mapped {len(norm_mapping)} norms: {list(norm_mapping.keys())}")

    # 6. Compute unified human alignment (stochastic mean + mode + det in one pass)
    print("\n6. Computing unified human alignment...")
    results = compute_human_alignment_unified(stoch_stats, det_df, human_df, norm_mapping)
    print(f"   Computed {len(results)} model-norm alignment values")

    if results.empty:
        print("ERROR: No correlations computed. Check data and mapping.")
        return

    # Print per-norm statistics
    print("\n   Coverage per norm:")
    for norm in sorted(results['norm'].unique()):
        norm_data = results[results['norm'] == norm]
        avg_n = norm_data['n_words'].mean()
        avg_r = norm_data['correlation'].mean()
        print(f"   {norm}: avg_n={avg_n:.0f}, avg_r={avg_r:.3f}")

    # 7. Fisher-z ranking
    print("\n7. Computing Fisher-z aggregate rankings...")
    ranking = compute_fisher_z_ranking(results)
    print("\n   Model Ranking by Average Human Correlation:")
    print(ranking.to_string(index=False))

    # 8. Compute mode vs temp0 alignment metrics
    print("\n8. Computing mode vs Temp 0 alignment metrics...")
    alignment = compute_mode_alignment_metrics(stoch_stats, det_df)
    print(f"   Computed alignment for {len(alignment)} (model, norm) pairs")

    # Stochastic advantage summary
    results['delta'] = results['correlation'] - results['temp0_human_corr']
    mean_delta = results['delta'].mean()
    n_pos_models = results.groupby('model')['delta'].mean().gt(0).sum()
    n_total_models = results['model'].nunique()
    n_pos_norms = results.groupby('norm')['delta'].mean().gt(0).sum()
    n_total_norms = results['norm'].nunique()
    print(f"\n   Stochastic advantage: Δr̄={mean_delta:.3f}; "
          f"{n_pos_models}/{n_total_models} models, "
          f"{n_pos_norms}/{n_total_norms} norms")

    # 9. Save results
    print("\n9. Saving results...")

    # Legacy-compatible outputs (for any downstream code that reads these)
    results[['model', 'norm', 'human_source', 'correlation', 'n_words', 'p_value']].to_csv(
        os.path.join(results_dir, 'human_correlation_results.csv'), index=False)
    ranking.to_csv(os.path.join(results_dir, 'human_correlation_ranking.csv'), index=False)

    results[['model', 'norm', 'mode_human_corr', 'mean_human_corr',
             'temp0_human_corr', 'kde_human_corr', 'n_words']].to_csv(
        os.path.join(results_dir, 'mode_human_alignment.csv'), index=False)

    if not alignment.empty:
        alignment.to_csv(os.path.join(results_dir, 'mode_alignment_metrics.csv'), index=False)

    stoch_stats.to_csv(os.path.join(results_dir, 'mode_stochastic_stats.csv'), index=False)

    # Clean dataset CSVs
    write_dataset_csvs(results, ranking, datasets_dir)

    # 10. Create visualizations
    print("\n10. Creating visualizations...")
    plot_correlation_heatmap(results, os.path.join(results_dir, 'human_correlation_heatmap.png'))
    plot_model_ranking(ranking, os.path.join(results_dir, 'human_correlation_ranking.png'))
    plot_stacked_by_norm(results, os.path.join(results_dir, 'human_correlation_by_norm.png'))
    if not results.empty:
        plot_human_alignment_comparison(results, os.path.join(results_dir, 'mode_human_comparison.png'))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")
    print(f"Datasets saved to: {datasets_dir}")


if __name__ == "__main__":
    main()
