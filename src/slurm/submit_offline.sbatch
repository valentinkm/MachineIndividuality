#!/bin/bash -l
#
# Psychometric Norm Generation — Combined det+stoch per model
#
#SBATCH -D ./
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err
#SBATCH -J psychnorms
#
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --mem-per-gpu=250000
#SBATCH --mail-type=none

# --- Configuration ---
CONTAINER="/dais/fs/scratch/vkriegmair/containers/vllm-openai_latest.sif"

if [ -z "$MODEL" ]; then
    echo "Error: MODEL not set. Submit with --export=MODEL=..."
    exit 1
fi

TP_SIZE=${TP_SIZE:-1}
MAX_RETRIES=${MAX_RETRIES:-3}
NORMS=${NORMS:-""}

# Conditions to run (default: both det and stoch)
# Override with e.g. CONDITIONS="det" to run only deterministic
CONDITIONS=${CONDITIONS:-"both"}

echo "══════════════════════════════════════════════════════════════"
echo "Psychnorms Generation — Combined Mode"
echo "Model:       $MODEL"
echo "TP:          $TP_SIZE"
echo "GPUs:        $SLURM_GPUS_ON_NODE"
echo "Max Retries: $MAX_RETRIES"
echo "Norms:       ${NORMS:-ALL}"
echo "Conditions:  $CONDITIONS"
echo "══════════════════════════════════════════════════════════════"

export TENSOR_PARALLEL_SIZE=$TP_SIZE

module purge
module load apptainer/1.4.1
APPTAINER=$(which apptainer 2>/dev/null || echo "/sw/dais/apptainer/1.4.1/bin/apptainer")

# Build optional norms flag
NORMS_FLAG=""
if [ -n "$NORMS" ]; then
    NORMS_FLAG="--norms $NORMS"
fi

run_condition() {
    local temp=$1
    local reps=$2
    local label=$3

    echo ""
    echo "── $label: temp=$temp, reps=$reps ──────────────────────────"
    echo "Start: $(date)"

    $APPTAINER exec --nv \
        --bind /dais/fs/scratch/vkriegmair/hf_cache:/dais/fs/scratch/vkriegmair/hf_cache \
        --env HF_TOKEN="${HF_TOKEN}" \
        --env HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
        --env HF_HOME="/dais/fs/scratch/vkriegmair/hf_cache" \
        $CONTAINER \
        bash -c "PYTHONPATH=src python3 -m psychnorms.generate \
            --model $MODEL \
            --temperature $temp \
            --repetitions $reps \
            --tensor-parallel-size $TP_SIZE \
            --max-retries-per-item $MAX_RETRIES \
            $NORMS_FLAG \
            --shuffle"

    echo "End:   $(date)"
    echo "── $label complete ──────────────────────────────────────────"
}

# Run conditions
if [ "$CONDITIONS" = "det" ] || [ "$CONDITIONS" = "both" ]; then
    run_condition 0.0 1 "DETERMINISTIC"
fi

if [ "$CONDITIONS" = "stoch" ] || [ "$CONDITIONS" = "both" ]; then
    run_condition 1.0 5 "STOCHASTIC"
fi

echo ""
echo "✅ All conditions complete for $MODEL"
