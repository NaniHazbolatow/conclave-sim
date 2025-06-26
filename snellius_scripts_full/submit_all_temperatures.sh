#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: bash submit_all_temperatures.sh <number>"
    echo "This will submit jobs for ALL temperature/rationality combinations"
    exit 1
fi

ARRAY_SIZE="$1"

echo "WARNING: This will submit jobs for ALL temperature/rationality combinations!"
echo "This means $(find temp_* -name "rat_*" -type d | wc -l) different parameter combinations."
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

for temp_dir in temp_*; do
    if [ -d "$temp_dir" ] && [ -f "$temp_dir/submit_all.sh" ]; then
        echo "Submitting all jobs in $temp_dir..."
        (cd "$temp_dir" && bash submit_all.sh "$ARRAY_SIZE")
        echo "Submitted $temp_dir, waiting 5 seconds before next batch..."
        sleep 5
    else
        echo "Skipping $temp_dir (not a directory or missing submit_all.sh)"
    fi
done

echo "All temperature directories submitted!"
