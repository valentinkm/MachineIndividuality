#!/bin/bash
# submit_all_models.sh
# Monitor with: squeue -u \$USER
# Check logs:   tail -f logs/slurm_*.out

set -e
cd /dais/fs/scratch/vkriegmair/GenPsychNorms

echo "Submitting Psych Norms Generation Jobs"
echo "=========================================="
echo ""

mkdir -p logs outputs/raw_behavior/model_norms

submit_job() {
    local model=$1
    local temp=$2
    local reps=$3
    local gpus=$4
    local tp=$5

    local cpus=$((12 * gpus))

    echo "  → $model @ temp=$temp reps=$reps (${gpus} GPU, TP=${tp})"

    if [ "$gpus" -eq 8 ]; then
        # Full exclusive node: partition=gpu, all memory, 96 CPUs
        sbatch --partition=gpu --gres=gpu:h200:8 --mem=0 \
            --cpus-per-task=96 --time=24:00:00 \
            --export=MODEL="$model",TEMP=$temp,REPEATS=$reps,TP_SIZE=$tp \
            src/slurm/submit_offline.sbatch
    elif [ "$gpus" -eq 1 ]; then
        sbatch --cpus-per-task=$cpus \
            --export=MODEL="$model",TEMP=$temp,REPEATS=$reps,TP_SIZE=$tp \
            src/slurm/submit_offline.sbatch
    else
        # Multi-GPU shared: override GPU count on gpu1 partition
        sbatch --gres=gpu:h200:$gpus --cpus-per-task=$cpus \
            --export=MODEL="$model",TEMP=$temp,REPEATS=$reps,TP_SIZE=$tp \
            src/slurm/submit_offline.sbatch
    fi
}

# --- Small models (1 GPU) ---
# temperature=0.0 / 1.0, repetitions=1 / 5, tensor-parallel-size=1, gpus=1
submit_job "phi_4" 0.0 1 1 1
submit_job "phi_4" 1.0 5 1 1
submit_job "granite_4_small" 0.0 1 1 1
submit_job "granite_4_small" 1.0 5 1 1

# --- Medium models (1 GPU) ---
# temperature=0.0 / 1.0, repetitions=1 / 5, tensor-parallel-size=1, gpus=1
submit_job "falcon_h1_34b_it" 0.0 1 1 1
submit_job "falcon_h1_34b_it" 1.0 5 1 1
submit_job "olmo3_32b_it" 0.0 1 1 1
submit_job "olmo3_32b_it" 1.0 5 1 1
submit_job "gptoss_20b" 0.0 1 1 1
submit_job "gptoss_20b" 1.0 5 1 1
submit_job "nomos_1" 0.0 1 1 1
submit_job "nomos_1" 1.0 5 1 1

# --- Large models (2 GPU, TP=2) ---
# temperature=0.0 / 1.0, repetitions=1 / 5, tensor-parallel-size=2, gpus=2
submit_job "qwen32b" 0.0 1 2 2
submit_job "qwen32b" 1.0 5 2 2
submit_job "mistral24b" 0.0 1 2 2
submit_job "mistral24b" 1.0 5 2 2
submit_job "gemma27b" 0.0 1 2 2
submit_job "gemma27b" 1.0 5 2 2
submit_job "gptoss_120b" 0.0 1 2 2
submit_job "gptoss_120b" 1.0 5 2 2

# --- Extra-large models (8 GPU, TP=8, full node) ---
# temperature=0.0 / 1.0, repetitions=1 / 5, tensor-parallel-size=8, gpus=8
submit_job "qwen3_235b_it" 0.0 1 8 8
submit_job "qwen3_235b_it" 1.0 5 8 8

echo ""
echo "All jobs submitted!"
echo ""
