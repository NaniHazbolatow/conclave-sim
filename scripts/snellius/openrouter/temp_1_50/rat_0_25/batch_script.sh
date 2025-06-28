#!/bin/bash

# This script is for running a single simulation locally.
# It extracts parameters from the directory structure.
# Usage: ./batch_script.sh [RUN_ID]
# If RUN_ID is not provided, it defaults to 1.

cd "$(dirname "$0")"

# ---- Set Run ID ----
RUN_ID=${1:-1}

# --- Dynamic Parameter Extraction ---
# Extracts temperature and rationality from the folder names
# e.g. .../temp_0_50/rat_0_75/batch_script.sh
CWD=$(pwd)
RAT_FOLDER_NAME=$(basename "$CWD")
TEMP_FOLDER_NAME=$(basename "$(dirname "$CWD")")
RATIONALITY=$(echo "$RAT_FOLDER_NAME" | cut -d'_' -f2- | sed 's/_/./')
TEMPERATURE=$(echo "$TEMP_FOLDER_NAME" | cut -d'_' -f2- | sed 's/_/./')

# ---- Setup Output Directory ----
OUTPUT_DIR="run_local_${RUN_ID}"
mkdir -p "$OUTPUT_DIR/logs"

echo "--- Starting Local Simulation ---"
echo "Run ID: ${RUN_ID}"
echo "Temperature: ${TEMPERATURE}"
echo "Rationality: ${RATIONALITY}"
echo "Output directory: ${OUTPUT_DIR}"
echo "---------------------------------"

# ---- Environment Variables ----
# Ensure your local Python environment is activated.
# For example: source path/to/your/venv/bin/activate
export TOKENIZERS_PARALLELISM=false
# Adjust OMP_NUM_THREADS based on your CPU cores if needed
export OMP_NUM_THREADS=8

source ../../conclave-sim/.venv/bin/activate

# ---- Run your simulation ----
echo "Starting Python simulation script..."
uv run ../../conclave-sim/simulations/run.py \
    --group xlarge \
    --temperature "$TEMPERATURE" \
    --rationality "$RATIONALITY" \
    --output-dir "$OUTPUT_DIR"

# ---- Finalization ----
sync
echo "--- Simulation Finished ---"
echo "Outputs for run ${RUN_ID} are in ${OUTPUT_DIR}"
echo "---------------------------"
