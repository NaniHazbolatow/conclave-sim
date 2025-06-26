#!/bin/bash

# Script to generate the entire snellius directory tree structure
# Usage: bash generate_tree.sh

# Define temperature and rationality values
TEMPERATURES=("0_10" "0_50" "1_00" "1_50" "2_00")
RATIONALITIES=("0_00" "0_25" "0_50" "0_75" "1_00")

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
        cp "$TEMPLATE_DIR/batch_script.sh" "$rat_dir/"
        
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

# Create a master submit_all.sh at the root level
cat > "submit_all_temperatures.sh" << 'EOF'
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
EOF

chmod +x "submit_all_temperatures.sh"

echo ""
echo "Directory tree generation complete!"
echo ""
echo "Generated structure using templates from: $TEMPLATE_DIR"
echo "├── temp_0_10/"
echo "│   ├── rat_0_00/ (submit_batch.sh, batch_script.sh)"
echo "│   ├── rat_0_25/ (submit_batch.sh, batch_script.sh)"
echo "│   ├── rat_0_50/ (submit_batch.sh, batch_script.sh)"
echo "│   ├── rat_0_75/ (submit_batch.sh, batch_script.sh)"
echo "│   ├── rat_1_00/ (submit_batch.sh, batch_script.sh)"
echo "│   └── submit_all.sh"
echo "├── temp_0_50/"
echo "│   └── ... (same structure)"
echo "├── temp_1_00/"
echo "├── temp_1_50/"
echo "├── temp_2_00/"
echo "└── submit_all_temperatures.sh"
echo ""
echo "Usage examples:"
echo "1. Submit 5 jobs for a specific temp/rat combination:"
echo "   cd temp_0_10/rat_0_25 && bash submit_batch.sh 5"
echo ""
echo "2. Submit 5 jobs for all rationality values at temp=0.10:"
echo "   cd temp_0_10 && bash submit_all.sh 5"
echo ""
echo "3. Submit 5 jobs for ALL temperature/rationality combinations:"
echo "   bash submit_all_temperatures.sh 5"
echo ""
echo "Note: All files are copied from the templates/ directory."
echo "      To modify the scripts, edit the template files and regenerate."
