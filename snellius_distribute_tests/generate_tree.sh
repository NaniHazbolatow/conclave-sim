#!/bin/bash

# Script to generate the entire snellius directory tree structure
# Usage: bash generate_tree.sh

# Define temperature and rationality values
TEMPERATURES=("0_99")
RATIONALITIES=("0_99")

# Base directory (current directory)
BASE_DIR=$(pwd)

# Template files directory
TEMPLATE_DIR="$BASE_DIR/templates"

# Check if template files exist
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "ERROR: Template directory not found at $TEMPLATE_DIR"
    echo "Please create a 'templates' directory with the following files:"
    echo "  - submit_batch.sh"
    echo "  - batch_script.sh"
    echo "  - submit_all.sh"
    exit 1
fi

REQUIRED_TEMPLATES=("submit_batch.sh" "batch_script.sh" "submit_all.sh")
for template in "${REQUIRED_TEMPLATES[@]}"; do
    if [ ! -f "$TEMPLATE_DIR/$template" ]; then
        echo "ERROR: Template file not found: $TEMPLATE_DIR/$template"
        exit 1
    fi
done

echo "Generating snellius directory tree structure..."
echo "Base directory: $BASE_DIR"
echo "Template directory: $TEMPLATE_DIR"

# Create directory structure and files
for temp in "${TEMPERATURES[@]}"; do
    temp_dir="temp_${temp}"
    echo "Creating temperature directory: $temp_dir"
    mkdir -p "$temp_dir"
    
    for rat in "${RATIONALITIES[@]}"; do
        rat_dir="$temp_dir/rat_${rat}"
        echo "  Creating rationality directory: $rat_dir"
        mkdir -p "$rat_dir"
        
        # Copy template files to rat directory
        cp "$TEMPLATE_DIR/submit_batch.sh" "$rat_dir/"
        
        # Copy and customize batch_script.sh with dynamic job name
        cp "$TEMPLATE_DIR/batch_script.sh" "$rat_dir/batch_script.sh"
        # Replace placeholders with actual temp and rat values (macOS compatible)
        sed -i '' "s/TEMP_PLACEHOLDER/temp_${temp}/g" "$rat_dir/batch_script.sh"
        sed -i '' "s/RAT_PLACEHOLDER/rat_${rat}/g" "$rat_dir/batch_script.sh"
        
        # Make scripts executable
        chmod +x "$rat_dir/submit_batch.sh"
        chmod +x "$rat_dir/batch_script.sh"
        
        echo "    Created files in $rat_dir"
    done
    
    # Copy submit_all.sh template to each temperature directory
    cp "$TEMPLATE_DIR/submit_all.sh" "$temp_dir/"
    chmod +x "$temp_dir/submit_all.sh"
    echo "  Created submit_all.sh in $temp_dir"
done

echo ""
echo "Directory tree generation complete!"
echo ""
echo "Generated structure using templates from: $TEMPLATE_DIR"
echo "├── temp_0_99/"
echo "│   └── rat_0_99/ (submit_batch.sh, batch_script.sh)"
echo "│   └── submit_all.sh"
echo ""
echo "Usage examples:"
echo "1. Submit 5 jobs for the temp/rat combination:"
echo "   cd temp_0_99/rat_0_99 && bash submit_batch.sh 5"
echo ""
echo "2. Submit 5 jobs for all rationality values at temp=0.99:"
echo "   cd temp_0_99 && bash submit_all.sh 5"
echo ""
echo "Note: All files are copied from the templates/ directory."
echo "      To modify the scripts, edit the template files and regenerate."
