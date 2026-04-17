# MachineIndividuality

**Replication code for:**  
Kriegmair, V. & Wulff, D. U. (2026). *Machine individuality: Separating genuine idiosyncrasy from response bias in large language models.*

## Overview

This repository generates psycholinguistic word ratings from 10 open-weight LLMs across 14 norms and analyzes them using crossed random-effects variance decomposition to separate shared consensus (Trait), systematic directional model bias (Bias), and stimulus-specific model individuality (Idiosyncrasy) from stochastic noise (Residual).

**Key result:** Model-specific idiosyncrasy accounts for 16.9% of variance on average (range: 4.8–34.0% across norms), robustly exceeding parametric null expectations. Cross-norm prediction analyses reveal these idiosyncrasies as coherent, model-specific semantic fingerprints (specificity ratios: 1.74–3.43).

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `src/psychnorms/` | Data generation: vLLM inference, prompt templates, model adapters |
| `src/analysis/` | Full analysis pipeline (LMM variance partitioning, simulation, extension, specificity) orchestrated by `run_pipeline_server.sh` |
| `src/slurm/` | SLURM job scripts for HPC submission |
| `data/` | Human psycholinguistic norms and metadata |
| `resources/` | Vocabulary list (107,083 cue words) and norm scale definitions |
| `supplementary/` | Supporting Information (Quarto source + rendered PDF) |
| `publication_plots/` | Final figures and tables (PDF) |
| `outputs/` | Generated data and analysis results (gitignored) |

## Models

| Model | Parameters | Organization |
|-------|-----------|-------------|
| Qwen3-32B | 32B | Alibaba |
| Qwen3-235B-A22B | 235B (22B active) | Alibaba |
| Mistral-Small-24B | 24B | Mistral AI |
| gemma-3-27b-it | 27B | Google |
| gpt-oss-20b | 20B | OpenAI |
| gpt-oss-120b | 120B | OpenAI |
| OLMo-3.1-32B | 32B | Allen AI |
| Falcon-H1-34B | 34B | TII |
| granite-4.0-h-small | ~8B | IBM |
| phi-4 | 14B | Microsoft |

## Norms (14)

All norms are drawn from the [psychNorms](https://arxiv.org/abs/2412.04936) metabase (Hussain et al., 2024).

| Norm | Source |
|------|--------|
| Arousal | Warriner, Kuperman & Brysbaert (2013). *Behavior Research Methods*, 45(4), 1191–1207. |
| Concreteness | Brysbaert, Warriner & Kuperman (2014). *Behavior Research Methods*, 46(3), 904–911. |
| Valence | Mohammad (2018). *Proc. ACL*, 174–184. |
| Visual | Lynott, Connell, Brysbaert, Brand & Carney (2020). *Behavior Research Methods*, 52(3), 1271–1291. |
| Auditory | Lynott et al. (2020). *Behavior Research Methods*, 52(3), 1271–1291. |
| Gustatory | Lynott et al. (2020). *Behavior Research Methods*, 52(3), 1271–1291. |
| Olfactory | Lynott et al. (2020). *Behavior Research Methods*, 52(3), 1271–1291. |
| Haptic | Lynott et al. (2020). *Behavior Research Methods*, 52(3), 1271–1291. |
| AoA (Kuperman) | Kuperman, Stadthagen-Gonzalez & Brysbaert (2012). *Behavior Research Methods*, 44(4), 978–990. |
| AoA (Brysbaert) | Brysbaert & Biemiller (2017). *Behavior Research Methods*, 49(4), 1520–1523. |
| Morality | Troché, Crutch & Reilly (2017). *Frontiers in Psychology*, 8, 1787. |
| Gender Association | Scott, Keitel, Becirspahic, Yao & Sereno (2019). *Behavior Research Methods*, 51(3), 1258–1270. |
| Humor | Engelthaler & Hills (2018). *Behavior Research Methods*, 50(3), 1116–1124. |
| Socialness | Diveica, Pexman & Binney (2023). *Behavior Research Methods*, 55(2), 461–473. |

## Replication

### Prerequisites

```bash
# Install environment
micromamba create -f environment.yml
micromamba activate mi_replication
```

### Step 1: Data Generation

> **Note:** This step requires a CUDA-capable GPU with vLLM installed. Pre-generated data is available upon request.

```bash
# Deterministic (T=0, 1 rep per cue×norm)
PYTHONPATH=. python -m psychnorms.generate --model qwen32b --temperature 0.0

# Stochastic (T=1.0, 5 reps per cue×norm)
PYTHONPATH=. python -m psychnorms.generate --model qwen32b --temperature 1.0 --repetitions 5

# Or submit all models via SLURM
./src/slurm/submit_all_models.sh
```

### Step 2: Analysis Pipeline

The full analysis pipeline is orchestrated by `src/analysis/run_pipeline_server.sh`:

```bash
# Full pipeline (Steps 0–8)
bash src/analysis/run_pipeline_server.sh

# Quick test (2 norms, 1000 words — verifies full pipeline in minutes)
bash src/analysis/run_pipeline_server.sh --test

# Resume from a specific step (e.g., skip postprocessing)
bash src/analysis/run_pipeline_server.sh --start 1

# Run only LMM core (skip extension/specificity/figure)
bash src/analysis/run_pipeline_server.sh --skip-ext
```

**Pipeline steps:**

| Step | Script | Description |
|------|--------|-------------|
| 0 | `postprocess_pipeline.py` | Clean raw CSVs → stochastic/deterministic splits |
| 1 | `LMM/01_prepare_arrow_shards.py` | Shard clean CSVs → per-norm Parquet files |
| 2 | `LMM/02_lmm_per_norm_full_server.R` | Fit crossed random-effects model per norm |
| 3 | `LMM/03_plot_variance_partitioning.R` + `04_extract_random_effects.R` | Merge per-norm results, extract BLUPs |
| 4 | `LMM/05_test_bias_consistency.R` | Test bias consistency across norms |
| 5 | `EXTENSION/01_human_alignment.py` | Human correlation + mode analysis |
| 6 | `SPECIFICITY/01_inter_norm_predictability.py` | Ridge regression specificity ratios |
| 7 | `LMM/06_combined_figure_publication.R` | Generate Figure 1 (Panel A + B) + summary tables |
| 8 | `SIMULATION/` | Parametric null simulation (N=100 bootstrap) |

### Step 3: Publication Figure

After the pipeline completes, the publication figure is at:
- `publication_plots/figure1_combined_short.pdf`

## Citation

```bibtex
@article{kriegmair2026machine,
  title={Machine individuality: Separating genuine idiosyncrasy from response bias in large language models},
  author={Kriegmair, Valentin and Wulff, Dirk U.},
  year={2026}
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
