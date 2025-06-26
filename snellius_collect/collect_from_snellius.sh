#!/bin/bash

# Script to collect results from different Snellius accounts
# Usage: bash collect_from_snellius.sh

# Define the account mappings using arrays (more compatible)
TEMP_FOLDERS=("temp_0_10" "temp_0_50" "temp_1_00" "temp_1_50" "temp_2_00")
USERNAMES=("spoulain" "spoulain1" "mbelkhatir" "kverlaan" "ehazbolatow")
PASSWORDS=("Wachtwoord123!" "Wachtwoord123!" "Myriam2001!!" "padkoh-pupteX-0fohzi" "Snelliusissnel123.")

# Function to get username for temperature folder
get_username() {
    local temp_folder="$1"
    case "$temp_folder" in
        "temp_0_10") echo "spoulain" ;;
        "temp_0_50") echo "spoulain1" ;;
        "temp_1_00") echo "mbelkhatir" ;;
        "temp_1_50") echo "kverlaan" ;;
        "temp_2_00") echo "ehazbolatow" ;;
        *) echo "" ;;
    esac
}

# Function to get password for username
get_password() {
    local username="$1"
    case "$username" in
        "spoulain") echo "Wachtwoord123!" ;;
        "spoulain1") echo "Wachtwoord123!" ;;
        "mbelkhatir") echo "Myriam2001!!" ;;
        "kverlaan") echo "padkoh-pupteX-0fohzi" ;;
        "ehazbolatow") echo "Snelliusissnel123." ;;
        *) echo "" ;;
    esac
}

# Base directory - go up to conclave-sim main folder and create collection directory
SCRIPT_DIR=$(pwd)
CONCLAVE_DIR=$(cd "$(dirname "$0")/.." && pwd)
COLLECTION_DIR="$CONCLAVE_DIR/snellius-collected"
SNELLIUS_HOST="snellius.surf.nl"

echo "Starting collection of results from Snellius accounts..."
echo "Script directory: $SCRIPT_DIR"
echo "Conclave directory: $CONCLAVE_DIR"
echo "Collection directory: $COLLECTION_DIR"
echo ""

# Create collection directory
mkdir -p "$COLLECTION_DIR"

# Function to collect results from account using sshpass
collect_from_account() {
    local temp_folder="$1"
    local username="$2"
    local password="$3"
    local remote_path="/home/$username/$temp_folder"
    local local_path="$COLLECTION_DIR/$temp_folder"
    
    echo "========================================="
    echo "Collecting $temp_folder from $username@$SNELLIUS_HOST"
    echo "Remote path: $remote_path"
    echo "Local path: $local_path"
    echo "========================================="
    
    # Check if sshpass is installed
    if ! command -v sshpass &> /dev/null; then
        echo "ERROR: sshpass is not installed. Please install it first:"
        echo "  macOS: brew install sshpass"
        echo "  Linux: apt-get install sshpass (or equivalent)"
        return 1
    fi
    
    # Test connection first
    echo "  Testing connection to $username@$SNELLIUS_HOST..."
    if sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" 2>/dev/null; then
        echo "  ✅ Connection test passed for $username"
    else
        echo "  ❌ Connection failed for $username - check username/password"
        return 1
    fi
    
    # Check if remote folder exists
    echo "  Checking if remote folder exists: $remote_path"
    if sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" \
        "[ -d '$remote_path' ] && echo 'Remote folder found'" 2>/dev/null; then
        echo "  ✅ Remote folder found for $username"
    else
        echo "  ⚠️  Remote folder not found for $username, skipping..."
        return 1
    fi
    
    # Create local directory
    mkdir -p "$local_path"
    
    # Copy the folder from remote to local
    echo "  Downloading $temp_folder for $username (this may take a while)..."
    if sshpass -p "$password" scp -r -o StrictHostKeyChecking=no \
        "$username@$SNELLIUS_HOST:$remote_path/*" "$local_path/" > /dev/null 2>&1; then
        echo "  ✅ Download completed successfully for $username"
        return 0
    else
        echo "  ❌ Download failed for $username"
        return 1
    fi
}

# Alternative function using SSH keys (more secure)
collect_from_account_with_keys() {
    local temp_folder="$1"
    local username="$2"
    local remote_path="/home/$username/$temp_folder"
    local local_path="$COLLECTION_DIR/$temp_folder"
    
    echo "========================================="
    echo "Collecting $temp_folder from $username@$SNELLIUS_HOST (using SSH keys)"
    echo "Remote: $remote_path → Local: $local_path"
    echo "========================================="
    
    # Check if remote folder exists
    if ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" "[ -d '$remote_path' ]" 2>/dev/null; then
        echo "  ✅ Remote folder found for $username"
    else
        echo "  ⚠️  Remote folder not found for $username, skipping..."
        return 1
    fi
    
    # Create local directory
    mkdir -p "$local_path"
    
    # Copy the folder
    echo "  Downloading $temp_folder for $username (this may take a while)..."
    if scp -r -o StrictHostKeyChecking=no \
        "$username@$SNELLIUS_HOST:$remote_path/*" "$local_path/" > /dev/null 2>&1; then
        echo "  ✅ Successfully collected $temp_folder from $username"
    else
        echo "  ❌ Failed to collect $temp_folder from $username"
        return 1
    fi
}

# Function to analyze results remotely without downloading
analyze_remotely() {
    local temp_folder="$1"
    local username="$2"
    local password="$3" # Optional, for sshpass
    local auth_type="$4" # "password" or "key"

    # echo "========================================="
    # echo "Analyzing $temp_folder on $username@$SNELLIUS_HOST"

    # Define the remote script as a here document.
    # Using a quoted EOF ('EOF') prevents any local variable expansion.
    # The script is passed to the remote shell, with username and temp_folder as arguments $1 and $2.
    read -r -d '' remote_script <<'EOF'
cd "/home/$1/$2" || { echo "ERROR: Could not cd to /home/$1/$2 on host $(hostname)"; exit 1; }

echo ''
echo "--------------------------------------------------------------------------------"
# Note: We use $2 and $1 for temp_folder and username passed as arguments
echo "Analysis for $2 on account $1"
printf "%-20s | %-10s | %-10s | %-10s | %s
" "Parameter Setting" "Submitted" "Ran" "Finished" "Status"
printf "%-20s | %-10s | %-10s | %-10s | %s
" "-------------------" "----------" "----------" "----------" "----------"

# Find all directories matching the rat_* pattern
find . -maxdepth 1 -type d -name "rat_*" | sort | while read RAT_DIR; do
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
                    SUCCESSFUL_RUN_DIRS=$(printf "%s
%s" "$SUCCESSFUL_RUN_DIRS" "$RUNDIR")
                fi

                # Now, check if this successful run is also a finished run
                if [ -f "$RUNDIR/results/simulation_summary.json" ]; then
                    if [ -z "$FINISHED_RUN_DIRS" ]; then
                        FINISHED_RUN_DIRS="$RUNDIR"
                    else
                        FINISHED_RUN_DIRS=$(printf "%s
%s" "$FINISHED_RUN_DIRS" "$RUNDIR")
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
        printf "%-20s | %-10s | %-10s | %-10s | %s
" "$DISPLAY_NAME" "$TOTAL_RUN_COUNT" "$COMPLETED_RUN_COUNT" "$FINISHED_RUN_COUNT" "$STATUS"
    )
done

echo "--------------------------------------------------------------------------------"
EOF

    # Execute based on auth type
    if [ "$auth_type" == "password" ]; then
        if ! command -v sshpass &> /dev/null; then
            echo "ERROR: sshpass is not installed. Please install it first."
            return 1
        fi
        if ! sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" > /dev/null 2>&1; then
            echo "  ❌ Connection failed for $username - check username/password"
            return 1
        fi
        # Pipe the script to the remote shell, passing username and temp_folder as arguments
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" 'bash -s' -- "$username" "$temp_folder" <<< "$remote_script"
    else # key-based
        if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" > /dev/null 2>&1; then
            echo "  ❌ Connection failed for $username - check SSH key setup"
            return 1
        fi
        # Pipe the script to the remote shell, passing username and temp_folder as arguments
        ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" 'bash -s' -- "$username" "$temp_folder" <<< "$remote_script"
    fi

    if [ $? -ne 0 ]; then
        echo "  ❌ Analysis failed for $username"
        return 1
    fi
    # echo "========================================="
    return 0
}

# Main menu
echo "What would you like to do?"
echo "1) Download all results from Snellius"
echo "2) Analyze results remotely on Snellius"
read -p "Enter your choice (1 or 2): " main_choice
echo ""

case $main_choice in
    1)
        # --- DOWNLOAD RESULTS ---
        echo "Choose authentication method for downloading:"
        echo "1) Use passwords (requires sshpass)"
        echo "2) Use SSH keys (more secure, requires key setup)"
        read -p "Enter your choice (1 or 2): " auth_method

        case $auth_method in
            1)
                echo "Using password authentication for download..."
                echo "Launching parallel downloads..."
                
                for temp_folder in "${TEMP_FOLDERS[@]}"; do
                    username=$(get_username "$temp_folder")
                    password=$(get_password "$username")
                    
                    if [ -z "$username" ] || [ -z "$password" ]; then
                        echo "⚠️  Warning: No username/password found for $temp_folder, skipping..."
                        continue
                    fi
                    
                    collect_from_account "$temp_folder" "$username" "$password" &
                done

                echo "Waiting for all downloads to complete..."
                wait
                echo "All parallel downloads have finished."
                ;;
                
            2)
                echo "Using SSH key authentication for download..."
                echo "Make sure you have SSH keys set up for all accounts!"
                echo "Launching parallel downloads..."
                
                for temp_folder in "${TEMP_FOLDERS[@]}"; do
                    username=$(get_username "$temp_folder")
                    
                    if [ -z "$username" ]; then
                        echo "⚠️  Warning: No username found for $temp_folder, skipping..."
                        continue
                    fi
                    
                    collect_from_account_with_keys "$temp_folder" "$username" &
                done

                echo "Waiting for all downloads to complete..."
                wait
                echo "All parallel downloads have finished."
                ;;
                
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac

        echo ""
        echo "Collection complete!"
        echo ""
        echo "Summary of collections:"
        for temp_folder in "${TEMP_FOLDERS[@]}"; do
            username=$(get_username "$temp_folder")
            if [ -n "$username" ]; then
                local_path="$COLLECTION_DIR/$temp_folder"
                # Check if directory exists before counting
                if [ -d "$local_path" ]; then
                    file_count=$(find "$local_path" -type f | wc -l | tr -d ' ')
                    echo "  $temp_folder ← $username@$SNELLIUS_HOST ($file_count files)"
                else
                    echo "  $temp_folder ← $username@$SNELLIUS_HOST (No files downloaded)"
                fi
            fi
        done
        echo ""
        echo "All collected files are in: $COLLECTION_DIR"
        echo ""
        echo "Next steps:"
        echo "1. Check the collected files in $COLLECTION_DIR"
        echo "2. Analyze your simulation results locally"
        ;;

    2)
        # --- ANALYZE REMOTELY ---
        echo "Choose authentication method for remote analysis:"
        echo "1) Use passwords (requires sshpass)"
        echo "2) Use SSH keys (more secure, requires key setup)"
        read -p "Enter your choice (1 or 2): " auth_method

        case $auth_method in
            1)
                echo "Using password authentication for remote analysis..."
                for temp_folder in "${TEMP_FOLDERS[@]}"; do
                    username=$(get_username "$temp_folder")
                    password=$(get_password "$username")
                    if [ -n "$username" ] && [ -n "$password" ]; then
                        analyze_remotely "$temp_folder" "$username" "$password" "password"
                    else
                        echo "⚠️  Skipping $temp_folder: username or password not found."
                    fi
                done
                ;;
            2)
                echo "Using SSH key authentication for remote analysis..."
                for temp_folder in "${TEMP_FOLDERS[@]}"; do
                    username=$(get_username "$temp_folder")
                    if [ -n "$username" ]; then
                        analyze_remotely "$temp_folder" "$username" "" "key"
                    else
                        echo "⚠️  Skipping $temp_folder: username not found."
                    fi
                done
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
        
        echo ""
        echo "Remote analysis complete."
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
