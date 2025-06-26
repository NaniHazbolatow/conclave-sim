#!/bin/bash
#
# renumber_runs.sh
#
# This script renumbers existing 'run' directories to be sequential 
# (e.g., run1, run2, run3...). It only modifies directories that need it.
#
# Usage: bash renumber_runs.sh
# Run this from a directory containing the temp_* folders (e.g., snellius-collected).

echo "Scanning for non-sequential runs to renumber..."

# Find all directories matching the temp_*/rat_* pattern
find . -type d -name "rat_*" | sort | while read RAT_DIR; do
    # Use a subshell to change directory safely
    (
        cd "$RAT_DIR" || exit

        # Get a list of run directories, sorted numerically.
        RUN_DIRS=$(find . -maxdepth 1 -type d -name "run[0-9]*" | sort -V)
        
        NEEDS_RENUMBERING=false
        CURRENT_INDEX=1
        for DIR in $RUN_DIRS; do
            DIR_NUM=$(echo "$DIR" | sed 's|./run||')
            if [ "$DIR_NUM" -ne "$CURRENT_INDEX" ]; then
                NEEDS_RENUMBERING=true
                break
            fi
            CURRENT_INDEX=$((CURRENT_INDEX + 1))
        done

        if [ "$NEEDS_RENUMBERING" = true ]; then
            echo "Renumbering runs in $RAT_DIR..."
            
            # Use a temporary suffix to avoid collisions during renaming
            TEMP_SUFFIX="_renaming_temp"

            # First pass: rename all to temporary names
            for DIR in $RUN_DIRS; do
                mv "$DIR" "${DIR}${TEMP_SUFFIX}"
            done

            # Second pass: rename from temporary to final sequential names
            FINAL_INDEX=1
            TEMP_DIRS=$(find . -maxdepth 1 -type d -name "run*[0-9]*${TEMP_SUFFIX}" | sort -V)
            for TEMP_DIR in $TEMP_DIRS; do
                mv "$TEMP_DIR" "./run${FINAL_INDEX}"
                FINAL_INDEX=$((FINAL_INDEX + 1))
            done
            echo "...done."
        fi
    )
done

echo "Renumbering finished."
