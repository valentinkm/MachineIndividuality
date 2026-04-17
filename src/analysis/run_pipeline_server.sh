#!/bin/bash
# ============================================================
# run_pipeline_server.sh
# Master orchestration script for MachineIndividuality full analysis.
#
# Reproduces all results for:
#   Kriegmair & Wulff, "Machine individuality: Separating genuine
#   idiosyncrasy from response bias in large language models" (PNAS).
#
# Pipeline Steps:
#   STEP 0  – Postprocessing: clean raw data → model_norms_clean/
#   STEP 1  – Arrow sharding: shard cleaned CSVs → per-norm Parquet
#   STEP 2  – LMM variance partitioning (parallel, per-norm)
#   STEP 3  – Merge per-norm LMM outputs + extract u_word / u_model
#   STEP 4  – Bias consistency analysis
#   STEP 5  – Extension analyses (human correlation · mode analysis)
#   STEP 6  – Specificity analysis (inter-norm predictability)
#   STEP 7  – Publication figure (combined Panel A + B)
#   STEP 8  – Parametric null simulation (parallel, per-norm)
#
# Usage:
#   # Full pipeline
#   bash run_pipeline_server.sh
#
#   # Quick test (2 norms, 1000 words, reduced workers)
#   bash run_pipeline_server.sh --test
#
#   # Resume from a specific step
#   bash run_pipeline_server.sh --start 3
#
#   # Run only steps 2-4
#   bash run_pipeline_server.sh --start 2 --stop 4
#
# Flags:
#   --start N     Start from step N (default: 0)
#   --stop  N     Stop after step N (default: 8)
#   --test        Test mode: 2 norms, 1000 words, reduced workers
#   --skip-lmm    Skip steps 1–4 (use existing LMM outputs)
#   --skip-sim    Skip step 8 (simulation)
#   --skip-ext    Skip steps 5–7 (only run LMM core)
#   --skip-pp     Skip step 0 (postprocessing; use existing clean data)
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
PYTHON_CMD="$(which python3)"

LMM_DIR="$PROJECT_ROOT/src/analysis/LMM"
SIM_DIR="$PROJECT_ROOT/src/analysis/SIMULATION"
EXT_DIR="$PROJECT_ROOT/src/analysis/EXTENSION"
SPEC_DIR="$PROJECT_ROOT/src/analysis/SPECIFICITY"
NORM_SCALES="$PROJECT_ROOT/resources/norm_scales.csv"
LMM_OUT="$PROJECT_ROOT/outputs/results/LMM_Full_filtered"
SIM_OUT="$PROJECT_ROOT/outputs/results/LMM_Simulation"

# Parse flags
START_STEP=0
STOP_STEP=8
SKIP_PP=false
SKIP_LMM=false
SKIP_SIM=false
SKIP_EXT=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --start)   START_STEP=$2; shift 2 ;;
        --stop)    STOP_STEP=$2; shift 2 ;;
        --skip-pp)  SKIP_PP=true; shift ;;
        --skip-lmm) SKIP_LMM=true; shift ;;
        --skip-sim) SKIP_SIM=true; shift ;;
        --skip-ext) SKIP_EXT=true; shift ;;
        --test)     TEST_MODE=true; shift ;;
        *)          echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Apply legacy skip flags to step ranges
if [ "$SKIP_PP" = true ]; then
    [ "$START_STEP" -eq 0 ] && START_STEP=1
fi
if [ "$SKIP_LMM" = true ]; then
    [ "$START_STEP" -le 1 ] && START_STEP=5
fi

cd "$PROJECT_ROOT"

# ── Test mode configuration ──
# In test mode: 2 norms, 1000 words, reduced parallelism
TEST_NORMS="concreteness_brysbaert arousal_warriner"
TEST_MAX_WORDS=1000
TEST_N_SIM=5
TEST_WORKERS=4

if [ "$TEST_MODE" = true ]; then
    LMM_WORKERS=2
    RE_WORKERS=2
    SIM_WORKERS=2
    SIM_CORES=1
    SIM_ITERS=$TEST_N_SIM
    EXT_WORKERS=$TEST_WORKERS
    SPEC_WORKERS=$TEST_WORKERS
    NORMS=$(echo "$TEST_NORMS" | tr ' ' '\n')
    ARROW_EXTRA_ARGS="--max-words $TEST_MAX_WORDS --norms $TEST_NORMS"
    EXT_EXTRA_ARGS="--sample $TEST_MAX_WORDS --workers $EXT_WORKERS"
    SPEC_EXTRA_ARGS="--sample_n $TEST_MAX_WORDS --workers $SPEC_WORKERS"
else
    LMM_WORKERS=15
    RE_WORKERS=14
    SIM_WORKERS=14
    SIM_CORES=2
    SIM_ITERS=100
    EXT_WORKERS=128
    SPEC_WORKERS=128
    # Read all norms from norm_scales.csv
    NORMS=$($PYTHON_CMD -c "\
import pandas as pd; \
print('\n'.join(pd.read_csv('$NORM_SCALES')['norm'].unique().tolist()))")
    ARROW_EXTRA_ARGS=""
    EXT_EXTRA_ARGS="--workers $EXT_WORKERS"
    SPEC_EXTRA_ARGS="--workers $SPEC_WORKERS"
fi

N_NORMS=$(echo "$NORMS" | wc -l | tr -d ' ')

echo ""
echo "============================================================"
echo "MachineIndividuality Full Analysis Pipeline"
echo "$(date)"
if [ "$TEST_MODE" = true ]; then
    echo "*** TEST MODE: $N_NORMS norms, $TEST_MAX_WORDS words ***"
fi
echo "Norms: $N_NORMS"
echo "Steps: ${START_STEP}–${STOP_STEP}"
echo "Flags: skip-pp=$SKIP_PP skip-lmm=$SKIP_LMM skip-sim=$SKIP_SIM skip-ext=$SKIP_EXT test=$TEST_MODE"
echo "============================================================"

run_step() {
    local step_num=$1
    local step_name=$2
    if [ "$step_num" -lt "$START_STEP" ] || [ "$step_num" -gt "$STOP_STEP" ]; then
        echo ">> Skipping STEP $step_num ($step_name) [outside range ${START_STEP}–${STOP_STEP}]"
        return 1
    fi
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP $step_num: $step_name"
    echo "Started: $(date)"
    echo "------------------------------------------------------------"
    return 0
}

# ============================================================
# STEP 0: Postprocessing (raw → clean CSVs)
# ============================================================
if run_step 0 "Postprocessing: raw → cleaned CSVs"; then
    $PYTHON_CMD "$REPO_ROOT/src/analysis/postprocess_pipeline.py"
    echo "✅ Postprocessing complete."
fi

# ============================================================
# STEP 1: Arrow sharding
# ============================================================
if run_step 1 "Shard cleaned CSVs into per-norm Parquet files"; then
    $PYTHON_CMD "$LMM_DIR/01_prepare_arrow_shards.py" $ARROW_EXTRA_ARGS
    echo "✅ Arrow sharding complete."
fi

# ============================================================
# STEP 2: Parallel LMM fitting
# ============================================================
if [ "$SKIP_LMM" = false ] && run_step 2 "LMM Variance Partitioning (parallel, $LMM_WORKERS workers)"; then
    echo "Norms: $(echo "$NORMS" | tr '\n' ' ')"
    mkdir -p "$LMM_OUT"

    echo "$NORMS" | xargs -n 1 -P $LMM_WORKERS -I {} bash -c '
        echo "  >> LMM: {}"
        Rscript "'$LMM_DIR'/02_lmm_per_norm_full_server.R" --norm "{}" --output_dir "'$LMM_OUT'" \
            > "'$LMM_OUT'/log_{}.txt" 2>&1 \
            && echo "  ✅ Done: {}" \
            || echo "  ❌ FAILED: {}  (see '$LMM_OUT'/log_{}.txt)"
    '
    echo "✅ All LMM runs completed."
fi

# ============================================================
# STEP 3: Merge LMM outputs + extract random effects
# ============================================================
if [ "$SKIP_LMM" = false ] && run_step 3 "Merge LMM shards + extract random effects"; then
    echo ">> 3a  Merge per-norm global/specific CSVs"
    Rscript "$LMM_DIR/03_plot_variance_partitioning.R" --output_dir "$LMM_OUT"

    echo ">> 3b  Extract u_word / u_model random effects (all norms)"
    echo "$NORMS" | xargs -n 1 -P $RE_WORKERS -I {} bash -c '
        echo "  >> Extracting RE: {}"
        Rscript "'$LMM_DIR'/04_extract_random_effects.R" --norm "{}" --output_dir "'$LMM_OUT'" \
            >> "'$LMM_OUT'/log_{}.txt" 2>&1 \
            && echo "  ✅ RE done: {}" \
            || echo "  ⚠️  RE failed: {} (non-fatal)"
    '
    echo "✅ LMM merge + RE extraction complete."
fi

# ============================================================
# STEP 4: Bias consistency analysis
# ============================================================
if [ "$SKIP_LMM" = false ] && run_step 4 "Bias Consistency Analysis"; then
    Rscript "$LMM_DIR/05_test_bias_consistency.R" --output_dir "$LMM_OUT"
    echo "✅ Bias consistency analysis complete."
fi

# ============================================================
# STEP 5: Extension analyses
# ============================================================
if [ "$SKIP_EXT" = false ] && run_step 5 "Extension Analyses"; then
    echo ">> 5  Human alignment analysis (correlation + mode, merged) ..."
    LMM_OUTPUT_DIR="$LMM_OUT" $PYTHON_CMD "$EXT_DIR/01_human_alignment.py" $EXT_EXTRA_ARGS
    echo "✅ Extension analyses complete."
fi

# ============================================================
# STEP 6: Specificity analysis
# ============================================================
if [ "$SKIP_EXT" = false ] && run_step 6 "Specificity Analysis"; then
    LMM_OUTPUT_DIR="$LMM_OUT" $PYTHON_CMD "$SPEC_DIR/01_inter_norm_predictability.py" $SPEC_EXTRA_ARGS
    echo "✅ Specificity analysis complete."
fi

# ============================================================
# STEP 7: Publication figure (combined Panel A + B)
# ============================================================
if [ "$SKIP_EXT" = false ] && run_step 7 "Publication Figure"; then
    Rscript "$LMM_DIR/06_combined_figure_publication.R"
    echo "✅ Publication figure generated."
fi

# ============================================================
# STEP 8: Parametric null simulation (last — most expensive)
# ============================================================
if [ "$SKIP_SIM" = false ] && run_step 8 "Parametric Null Simulation"; then
    echo ">> 8a  Prepare simulation data ..."
    $PYTHON_CMD "$SIM_DIR/00_prepare_simulation_data.py"

    echo ">> 8b  Run simulation ($N_NORMS norms × $SIM_CORES cores each, $SIM_ITERS iterations) ..."
    mkdir -p "$SIM_OUT/logs"

    echo "$NORMS" | xargs -n 1 -P $SIM_WORKERS -I {} bash -c '
        echo "  >> Simulation: {}"
        LMM_OUTPUT_DIR="'$LMM_OUT'" Rscript "'$SIM_DIR'/01_parametric_null_simulation.R" \
            --norm "{}" --n_sim '$SIM_ITERS' --n_cores '$SIM_CORES' \
            --input_dir "outputs/raw_behavior/model_norms_clean/stochastic" \
            --nrows -1 \
            > "'$SIM_OUT'/logs/{}.log" 2>&1 \
            && echo "  ✅ Sim done: {}" \
            || echo "  ❌ Sim FAILED: {} (see '$SIM_OUT'/logs/{}.log)"
    '
    echo "✅ All simulations completed."

    echo ">> 8c  Aggregate simulation results ..."
    LMM_OUTPUT_DIR="$LMM_OUT" Rscript "$SIM_DIR/02_aggregate_simulation_results.R"
    LMM_OUTPUT_DIR="$LMM_OUT" Rscript "$SIM_DIR/02_aggregate_model_specific_results.R"
    echo "✅ Simulation aggregation complete."
fi

echo ""
echo "============================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "Steps executed: ${START_STEP}–${STOP_STEP}"
if [ "$TEST_MODE" = true ]; then
    echo "Mode: TEST ($N_NORMS norms, $TEST_MAX_WORDS words)"
fi
echo "$(date)"
echo "============================================================"
