#!/bin/bash

# Script to submit a batch of jobs to SLURM
# Usage: bash submit_batch.sh <array_size>

if [ -z "$1" ]; then
    echo "Usage: bash submit_new_batch.sh <array_size>"
    exit 1
fi
ARRAY_SIZE="$1"

LAST_RUN=$(find . -maxdepth 1 -type d -regex './run[0-9]+' | sed 's|./run||' | sort -n | tail -n1)
if [[ -z "$LAST_RUN" ]]; then
    LAST_RUN=0
fi

BATCH_START=$((LAST_RUN + 1))
echo "Found existing runs: $(find . -maxdepth 1 -type d -regex './run[0-9]+' | sed 's|./run||' | sort -n | tr '\n' ' ')"
echo "LAST_RUN detected: $LAST_RUN"
echo "Launching array with BATCH_START_RUN = $BATCH_START"

for ((i=0; i<ARRAY_SIZE; i++)); do
    RUN_NUM=$((BATCH_START + i))
    mkdir -p "run${RUN_NUM}/snellius_logs"
done

export BATCH_START_RUN=$BATCH_START

# Launch job array, logging directly to the correct folder per array task
# Use array starting from BATCH_START to match run directory numbers
jobid=$(sbatch \
    --export=ALL,BATCH_START_RUN \
    --array=$BATCH_START-$((BATCH_START+ARRAY_SIZE-1)) \
    --output=run%a/snellius_logs/run_%A_%a.out \
    --error=run%a/snellius_logs/run_%A_%a.err \
    batch_script.sh | awk '{print $4}')
echo "Submitted as job $jobid"
echo "Logs will be written directly to run<N>/snellius_logs/. No post-processing required!"
