#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: bash submit_all.sh <number>"
    exit 1
fi

ARRAY_SIZE="$1"

for dir in rat_*; do
    if [ -d "$dir" ] && [ -f "$dir/submit_batch.sh" ]; then
        echo "Submitting in $dir..."
        (cd "$dir" && sbatch submit_batch.sh "$ARRAY_SIZE")
    else
        echo "Skipping $dir (not a directory or missing submit_batch.sh)"
    fi
done
