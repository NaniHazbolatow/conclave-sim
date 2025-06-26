#!/bin/bash
#
# analyze_runs.sh
#
# This script reports the total, completed, and finished 'run' directories for each 
# parameter setting and checks if the completed runs are sequentially numbered.
# - A completed run is a directory that contains more than just 'snellius_logs'.
# - A finished run is a completed run that also contains 'results/simulation_summary.json'.
# It does NOT modify any files.
#
# Usage: bash analyze_runs.sh
# Run this from a directory containing the temp_* folders (e.g., snellius-collected).

echo "Analyzing Run Status..."

# Print table header
echo "===================================================================================================="
printf "%-20s | %-10s | %-10s | %-10s | %s\n" "Parameter Setting" "Submitted" "Ran" "Finished" "Status"
printf "%-20s | %-10s | %-10s | %-10s | %s\n" "-------------------" "----------" "----------" "----------" "----------"

LAST_TEMP_DIR=""

# Find all directories matching the temp_*/rat_* pattern
find . -type d -name "rat_*" | sort | while read RAT_DIR; do
    # Extract the parent temp_* directory from a path like ./temp_0_10/rat_0_00
    CURRENT_TEMP_DIR=$(echo "$RAT_DIR" | cut -d'/' -f2)

    # Check if the temp directory has changed and it's not the first iteration
    if [ -n "$LAST_TEMP_DIR" ] && [ "$CURRENT_TEMP_DIR" != "$LAST_TEMP_DIR" ]; then
        printf "%-20s | %-10s | %-10s | %-10s | %s\n" "-------------------" "----------" "----------" "----------" "----------"
    fi

    # Use a subshell to change directory safely
    (
        cd "$RAT_DIR" || exit

        # Get a list of all run directories, sorted numerically.
        ALL_RUN_DIRS=$(find . -maxdepth 1 -type d -name "run[0-9]*" | sort -V)
        
        if [ -z "$ALL_RUN_DIRS" ]; then
            TOTAL_RUN_COUNT=0
        else
            TOTAL_RUN_COUNT=$(echo "$ALL_RUN_DIRS" | wc -l | tr -d ' ')
        fi

        SUCCESSFUL_RUN_DIRS=""
        FINISHED_RUN_DIRS=""
        for RUNDIR in $ALL_RUN_DIRS; do
            # Count items inside the run directory, ignoring .DS_Store on macOS
            NUM_ITEMS=$(find "$RUNDIR" -mindepth 1 -maxdepth 1 -not -name ".DS_Store" | wc -l)
            
            IS_FAILED=false
            # A run is considered failed if it's empty or contains ONLY the snellius_logs directory.
            if [ "$NUM_ITEMS" -eq 0 ]; then
                IS_FAILED=true
            elif [ "$NUM_ITEMS" -eq 1 ] && [ -d "$RUNDIR/snellius_logs" ]; then
                IS_FAILED=true
            fi

            if [ "$IS_FAILED" = false ]; then
                # It's a successful run, add it to our list.
                if [ -z "$SUCCESSFUL_RUN_DIRS" ]; then
                    SUCCESSFUL_RUN_DIRS="$RUNDIR"
                else
                    SUCCESSFUL_RUN_DIRS=$(printf "%s\n%s" "$SUCCESSFUL_RUN_DIRS" "$RUNDIR")
                fi

                # Now, check if this successful run is also a finished run
                if [ -f "$RUNDIR/results/simulation_summary.json" ]; then
                    if [ -z "$FINISHED_RUN_DIRS" ]; then
                        FINISHED_RUN_DIRS="$RUNDIR"
                    else
                        FINISHED_RUN_DIRS=$(printf "%s\n%s" "$FINISHED_RUN_DIRS" "$RUNDIR")
                    fi
                fi
            fi
        done

        # Count the successful runs
        if [ -z "$SUCCESSFUL_RUN_DIRS" ]; then
            COMPLETED_RUN_COUNT=0
        else
            COMPLETED_RUN_COUNT=$(echo "$SUCCESSFUL_RUN_DIRS" | wc -l | tr -d ' ')
        fi

        # Count the finished runs
        if [ -z "$FINISHED_RUN_DIRS" ]; then
            FINISHED_RUN_COUNT=0
        else
            FINISHED_RUN_COUNT=$(echo "$FINISHED_RUN_DIRS" | wc -l | tr -d ' ')
        fi

        # Check if the successful runs are sequentially numbered
        NEEDS_RENUMBERING=false
        if [ "$COMPLETED_RUN_COUNT" -gt 0 ]; then
            CURRENT_INDEX=1
            for DIR in $SUCCESSFUL_RUN_DIRS; do
                # Extract number from directory name like './run12' -> '12'
                DIR_NUM=$(echo "$DIR" | sed 's|./run||')
                if [ "$DIR_NUM" -ne "$CURRENT_INDEX" ]; then
                    NEEDS_RENUMBERING=true
                    break
                fi
                CURRENT_INDEX=$((CURRENT_INDEX + 1))
            done
        fi

        STATUS="Sequential"
        if [ "$NEEDS_RENUMBERING" = true ]; then
            STATUS="Needs Renumbering"
        fi

        # Clean up the directory name for display
        DISPLAY_NAME=$(echo "$RAT_DIR" | sed 's|./||')
        printf "%-20s | %-10s | %-10s | %-10s | %s\n" "$DISPLAY_NAME" "$TOTAL_RUN_COUNT" "$COMPLETED_RUN_COUNT" "$FINISHED_RUN_COUNT" "$STATUS"
    )

    # Update the last temp directory for the next iteration
    LAST_TEMP_DIR=$CURRENT_TEMP_DIR
done

echo "===================================================================================================="
echo "Analysis finished. To fix numbering of successful runs, run renumber_runs.sh"
