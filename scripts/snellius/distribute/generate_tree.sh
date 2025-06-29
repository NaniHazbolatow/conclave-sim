#!/bin/bash

###############################################################################
# generate_tree.sh
#
# Script to generate the Snellius simulation directory tree structure.
# Usage:
#   bash generate_tree.sh
#
# CONFIGURATION:
# - Edit TEMPERATURES, RATIONALITIES, and TEMPLATE_DIR below,
#   or export them as environment variables before running.
#
# EXAMPLES:
#   To set custom values temporarily:
#     export TEMPERATURES="0_10 0_50"
#     export RATIONALITIES="0_00 0_50 1_00"
#     export TEMPLATE_DIR="/path/to/templates"
#     bash generate_tree.sh
#
# All job/cleanup scripts are copied from the templates directory.
#
# Never commit secrets to version control.
###############################################################################

# --- CONFIGURATION ---

# Read from env or use defaults
: "${TEMPERATURES:="0_10 0_50 1_00 1_50 2_00"}"
: "${RATIONALITIES:="0_00 0_25 0_50 0_75 1_00"}"
: "${TEMPLATE_DIR:="$PWD/templates"}"

# --- END CONFIGURATION ---

# Convert config variables into bash arrays
IFS=' ' read -r -a TEMPERATURES_ARR <<< "$TEMPERATURES"
IFS=' ' read -r -a RATIONALITIES_ARR <<< "$RATIONALITIES"

# Required templates
REQUIRED_TEMPLATES=("submit_batch.sh" "batch_script.sh" "submit_all.sh" "cleanup_failed_runs.sh" "cleanup_all.sh")

# Check if template files exist
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "ERROR: Template directory not found at $TEMPLATE_DIR"
    echo "Please create a 'templates' directory with the following files:"
    printf "  - %s\n" "${REQUIRED_TEMPLATES[@]}"
    exit 1
fi

for template in "${REQUIRED_TEMPLATES[@]}"; do
    if [ ! -f "$TEMPLATE_DIR/$template" ]; then
        echo "ERROR: Template file not found: $TEMPLATE_DIR/$template"
        exit 1
    fi
done

echo "Generating snellius directory tree structure..."
echo "Base directory: $PWD"
echo "Template directory: $TEMPLATE_DIR"
echo "Temperatures: ${TEMPERATURES_ARR[*]}"
echo "Rationalities: ${RATIONALITIES_ARR[*]}"

for temp in "${TEMPERATURES_ARR[@]}"; do
    temp_dir="temp_${temp}"
    echo "Creating temperature directory: $temp_dir"
    mkdir -p "$temp_dir"

    for rat in "${RATIONALITIES_ARR[@]}"; do
        rat_dir="$temp_dir/rat_${rat}"
        echo "  Creating rationality directory: $rat_dir"
        mkdir -p "$rat_dir"

        # Copy required files
        cp "$TEMPLATE_DIR/submit_batch.sh" "$rat_dir/"
        cp "$TEMPLATE_DIR/cleanup_failed_runs.sh" "$rat_dir/"

        # Copy and customize batch_script.sh
        cp "$TEMPLATE_DIR/batch_script.sh" "$rat_dir/batch_script.sh"
        # Replace placeholders with actual values (macOS and GNU compatible)
        if sed --version >/dev/null 2>&1; then
            # GNU sed (Linux)
            sed -i "s/TEMP_PLACEHOLDER/temp_${temp}/g" "$rat_dir/batch_script.sh"
            sed -i "s/RAT_PLACEHOLDER/rat_${rat}/g" "$rat_dir/batch_script.sh"
        else
            # macOS BSD sed
            sed -i '' "s/TEMP_PLACEHOLDER/temp_${temp}/g" "$rat_dir/batch_script.sh"
            sed -i '' "s/RAT_PLACEHOLDER/rat_${rat}/g" "$rat_dir/batch_script.sh"
        fi

        # Make scripts executable
        chmod +x "$rat_dir/submit_batch.sh" "$rat_dir/batch_script.sh" "$rat_dir/cleanup_failed_runs.sh"

        echo "    Created files in $rat_dir"
    done

    # Copy submit_all.sh and cleanup_all.sh to each temperature directory
    cp "$TEMPLATE_DIR/submit_all.sh" "$temp_dir/"
    cp "$TEMPLATE_DIR/cleanup_all.sh" "$temp_dir/"
    chmod +x "$temp_dir/submit_all.sh" "$temp_dir/cleanup_all.sh"
    echo "  Created submit_all.sh and cleanup_all.sh in $temp_dir"
done

echo ""
echo "Directory tree generation complete!"
echo ""
echo "Generated structure using templates from: $TEMPLATE_DIR"
for temp in "${TEMPERATURES_ARR[@]}"; do
    echo "├── temp_${temp}/"
    for rat in "${RATIONALITIES_ARR[@]}"; do
        echo "│   ├── rat_${rat}/ (submit_batch.sh, batch_script.sh)"
    done
    echo "│   ├── submit_all.sh"
    echo "│   └── cleanup_all.sh"
done
echo ""
echo "Usage examples:"
echo "1. Submit 5 jobs for a specific temp/rat combination:"
echo "   cd temp_${TEMPERATURES_ARR[0]}/rat_${RATIONALITIES_ARR[1]} && bash submit_batch.sh 5"
echo ""
echo "2. Submit 5 jobs for all rationality values at a given temp:"
echo "   cd temp_${TEMPERATURES_ARR[0]} && bash submit_all.sh 5"
echo ""
echo "Note: All files are copied from the templates/ directory."
echo "      To modify the scripts, edit the template files and regenerate."

