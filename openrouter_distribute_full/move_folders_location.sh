#!/bin/bash

# Script to copy all temp_* folders from openrouter_distribute_full to outside conclave-sim

# Get the absolute path to the current script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the parent directory of conclave-sim
TARGET_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Copy each temp_* folder
for folder in "$SCRIPT_DIR"/temp_*; do
    if [ -d "$folder" ]; then
        echo "Copying $(basename "$folder") to $TARGET_DIR"
        cp -R "$folder" "$TARGET_DIR"
    fi
done

echo "All temp_* folders copied to $TARGET_DIR"