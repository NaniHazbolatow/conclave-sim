#!/bin/bash
#
# cleanup_failed_runs.sh
#
# This script removes run directories that only contain the snellius_logs/ folder.
# This state typically occurs when a SLURM job is submitted but fails or is
# cancelled before any simulation results are generated.
#
# Usage: bash cleanup_failed_runs.sh
# Run this from a directory containing run folders (e.g., temp_*/rat_*/).

echo "Scanning for failed or cancelled run directories..."

# Find directories named 'run*' at the current level
find . -maxdepth 1 -type d -name "run[0-9]*" | while read RUNDIR; do
    # Count the number of items in the directory, ignoring macOS-specific .DS_Store files.
    NUM_ITEMS=$(find "$RUNDIR" -mindepth 1 -maxdepth 1 -not -name ".DS_Store" | wc -l)

    # Check if there is exactly one item left after ignoring .DS_Store
    if [ "$NUM_ITEMS" -eq 1 ]; then
        # And if that single item is the 'snellius_logs' directory
        if [ -d "$RUNDIR/snellius_logs" ]; then
            echo "Found incomplete run: $RUNDIR. Removing..."
            rm -rf "$RUNDIR"
        fi
    fi
done

echo "Cleanup finished."
