#!/bin/bash
#SBATCH --job-name=temp_2_00_rat_0_75
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=run_%a/snellius_logs/run_%A_%a.out
#SBATCH --error=run_%a/snellius_logs/run_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=salome.poulain@student.uva.nl

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")"

# ---- Calculate unique run number ----
if [ -z "$BATCH_START_RUN" ]; then
    echo "ERROR: BATCH_START_RUN not set! You must launch via get_next_run.sh"
    exit 1
fi


# --- Dynamic Parameter Extraction ---
CWD=$(pwd)
RAT_FOLDER_NAME=$(basename "$CWD")
TEMP_FOLDER_NAME=$(basename $(dirname "$CWD"))
RATIONALITY=$(echo "$RAT_FOLDER_NAME" | cut -d'_' -f2- | sed 's/_/./')
TEMPERATURE=$(echo "$TEMP_FOLDER_NAME" | cut -d'_' -f2- | sed 's/_/./')

# Debug: Show what we're calculating
echo "DEBUG: BATCH_START_RUN = $BATCH_START_RUN"
echo "DEBUG: SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"

# Fixed calculation: SLURM_ARRAY_TASK_ID IS the run number when array starts from BATCH_START
THIS_RUN_NUM=$SLURM_ARRAY_TASK_ID
OUTPUT_DIR="run${THIS_RUN_NUM}"

echo "DEBUG: Using THIS_RUN_NUM = $THIS_RUN_NUM (directly from SLURM_ARRAY_TASK_ID)"
echo "DEBUG: OUTPUT_DIR = $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/snellius_logs"

# ---- Load modules and set env variables (unchanged) ----
module purge
module load 2023
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# ---- Start GPU monitoring in background ----
GPU_MONITOR_LOG="$OUTPUT_DIR/snellius_logs/gpu_monitor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
GPU_INITIAL_LOG="$OUTPUT_DIR/snellius_logs/gpu_initial_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
GPU_FINAL_LOG="$OUTPUT_DIR/snellius_logs/gpu_final_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw --format=csv --loop=10 > "$GPU_MONITOR_LOG" &
MONITOR_PID=$!
nvidia-smi > "$GPU_INITIAL_LOG"

# ---- Run your simulation ----
echo "Starting simulation for run $THIS_RUN_NUM"
python ../../conclave-sim/simulations/run.py \
    --group xlarge \
    --temperature "$TEMPERATURE" \
    --rationality "$RATIONALITY" \
    --output-dir "$OUTPUT_DIR"

# ---- Stop GPU monitoring ----
kill $MONITOR_PID
nvidia-smi > "$GPU_FINAL_LOG"

sync
echo "Outputs for run $THIS_RUN_NUM are in $OUTPUT_DIR"
echo "Outputs for run $THIS_RUN_NUM are in $OUTPUT_DIR"
