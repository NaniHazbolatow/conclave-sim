#!/bin/bash

# Script to run a batch of local simulations sequentially.
# Usage: bash submit_batch.sh <num_runs>

if [ -z "$1" ]; then
    echo "Usage: bash submit_batch.sh <num_runs>"
    exit 1
fi
NUM_RUNS="$1"

# Find the last local run number to avoid overwriting
LAST_RUN=$(find . -maxdepth 1 -type d -name 'run_local_*' | sed 's|./run_local_||' | sort -n | tail -n1)
if [[ -z "$LAST_RUN" ]]; then
    LAST_RUN=0
fi

START_RUN_ID=$((LAST_RUN + 1))
END_RUN_ID=$((START_RUN_ID + NUM_RUNS - 1))

echo "Last local run detected: $LAST_RUN"
echo "Starting new runs from $START_RUN_ID to $END_RUN_ID"

for ((i=START_RUN_ID; i<=END_RUN_ID; i++)); do
    echo ""
    echo "--- Starting local run with ID: $i ---"
    bash batch_script.sh "$i"
    echo "--- Finished local run with ID: $i ---"
done

echo ""
echo "All $NUM_RUNS local simulations finished."
