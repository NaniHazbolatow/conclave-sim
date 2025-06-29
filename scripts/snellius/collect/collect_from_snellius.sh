#!/bin/bash

##########################################################################
# Script: collect_from_snellius.sh
#
# Collect results from multiple Snellius accounts without hardcoded secrets.
# USAGE:
#   1. Copy this script.
#   2. Create a .env file in the same directory:
#        SNELLIUS_PASS_user1="password1"
#        SNELLIUS_PASS_user2="password2"
#        ...
#      (Make sure .env is in .gitignore!)
#   3. Run: bash collect_from_snellius.sh
#
# NOTE: Never commit real passwords or .env to git!
##########################################################################

# Load environment variables from .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Define temp folders and usernames (edit these for your use case)
TEMP_FOLDERS=("temp_0_10" "temp_0_50" "temp_1_00" "temp_1_50" "temp_2_00")
USERNAMES=("user1" "user2" "user3" "user4" "user5")  # Replace with your actual usernames

# Mapping temp folders to usernames (update as needed)
get_username() {
    local temp_folder="$1"
    case "$temp_folder" in
        "temp_0_10") echo "user1" ;;
        "temp_0_50") echo "user2" ;;
        "temp_1_00") echo "user3" ;;
        "temp_1_50") echo "user4" ;;
        "temp_2_00") echo "user5" ;;
        *) echo "" ;;
    esac
}

# Fetch password from environment variables (never hardcoded!)
get_password() {
    local username="$1"
    eval echo "\$SNELLIUS_PASS_${username}"
}

# Setup paths
SCRIPT_DIR=$(pwd)
CONCLAVE_DIR=$(cd "$(dirname "$0")/.." && pwd)
COLLECTION_DIR="$CONCLAVE_DIR/snellius-collected"
SNELLIUS_HOST="snellius.surf.nl"

echo "Starting collection of results from Snellius accounts..."
echo "Script directory: $SCRIPT_DIR"
echo "Conclave directory: $CONCLAVE_DIR"
echo "Collection directory: $COLLECTION_DIR"
echo ""

mkdir -p "$COLLECTION_DIR"

# Download with sshpass (password auth)
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
    
    if ! command -v sshpass &> /dev/null; then
        echo "ERROR: sshpass is not installed."
        return 1
    fi
    
    echo "  Testing connection to $username@$SNELLIUS_HOST..."
    if ! sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" 2>/dev/null; then
        echo "  ❌ Connection failed for $username"
        return 1
    fi
    
    echo "  Checking if remote folder exists: $remote_path"
    if ! sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" "[ -d '$remote_path' ] && echo 'Remote folder found'" 2>/dev/null; then
        echo "  ⚠️  Remote folder not found for $username, skipping..."
        return 1
    fi

    mkdir -p "$local_path"
    echo "  Downloading $temp_folder for $username ..."
    if sshpass -p "$password" scp -r -o StrictHostKeyChecking=no \
        "$username@$SNELLIUS_HOST:$remote_path/*" "$local_path/" > /dev/null 2>&1; then
        echo "  ✅ Download completed successfully for $username"
        return 0
    else
        echo "  ❌ Download failed for $username"
        return 1
    fi
}

# Download with SSH keys
collect_from_account_with_keys() {
    local temp_folder="$1"
    local username="$2"
    local remote_path="/home/$username/$temp_folder"
    local local_path="$COLLECTION_DIR/$temp_folder"
    
    echo "========================================="
    echo "Collecting $temp_folder from $username@$SNELLIUS_HOST (using SSH keys)"
    echo "Remote: $remote_path → Local: $local_path"
    echo "========================================="
    
    if ! ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" "[ -d '$remote_path' ]" 2>/dev/null; then
        echo "  ⚠️  Remote folder not found for $username, skipping..."
        return 1
    fi
    
    mkdir -p "$local_path"
    echo "  Downloading $temp_folder for $username ..."
    if scp -r -o StrictHostKeyChecking=no \
        "$username@$SNELLIUS_HOST:$remote_path/*" "$local_path/" > /dev/null 2>&1; then
        echo "  ✅ Successfully collected $temp_folder from $username"
    else
        echo "  ❌ Failed to collect $temp_folder from $username"
        return 1
    fi
}

# Analyze remotely
analyze_remotely() {
    local username="$1"
    local password="$2"
    local auth_type="$3" # "password" or "key"
    SNELLIUS_HOST="snellius.surf.nl"

    read -r -d '' remote_script <<-'EOF'
cd "/home/$1" || { echo "ERROR: Could not cd to /home/$1 on host $(hostname)"; exit 1; }
find . -maxdepth 1 -type d -name "temp_*" | sort | while read TEMP_DIR; do
    (
        cd "$TEMP_DIR" || exit
        TEMP_NAME=$(basename "$TEMP_DIR")
        echo ""
        echo "--------------------------------------------------------------------------------"
        echo "Analysis for $TEMP_NAME on account $1"
        printf "%-20s | %10s | %10s | %10s | %s\n" "Parameter Setting" "Submitted" "Ran" "Finished" "Status"
        printf "%-20s | %10s | %10s | %10s | %s\n" "-------------------" "----------" "----------" "----------" "----------"
        find . -maxdepth 1 -type d -name "rat_*" | sort | while read RAT_DIR; do
            (
                cd "$RAT_DIR" || exit
                ALL_RUN_DIRS=$(find . -maxdepth 1 -type d -name "run[0-9]*" | sort -V)
                if [ -z "$ALL_RUN_DIRS" ]; then
                    TOTAL_RUN_COUNT=0
                else
                    TOTAL_RUN_COUNT=$(echo "$ALL_RUN_DIRS" | wc -l | tr -d ' ')
                fi
                SUCCESSFUL_RUN_DIRS=""
                FINISHED_RUN_DIRS=""
                for RUNDIR in $ALL_RUN_DIRS; do
                    NUM_ITEMS=$(find "$RUNDIR" -mindepth 1 -maxdepth 1 -not -name ".DS_Store" | wc -l)
                    IS_FAILED=false
                    if [ "$NUM_ITEMS" -eq 0 ]; then IS_FAILED=true; fi
                    if [ "$NUM_ITEMS" -eq 1 ] && [ -d "$RUNDIR/snellius_logs" ]; then IS_FAILED=true; fi
                    if [ "$IS_FAILED" = false ]; then
                        if [ -z "$SUCCESSFUL_RUN_DIRS" ]; then
                            SUCCESSFUL_RUN_DIRS="$RUNDIR"
                        else
                            SUCCESSFUL_RUN_DIRS=$(printf "%s\n%s" "$SUCCESSFUL_RUN_DIRS" "$RUNDIR")
                        fi
                        if [ -f "$RUNDIR/results/simulation_summary.json" ]; then
                            if [ -z "$FINISHED_RUN_DIRS" ]; then
                                FINISHED_RUN_DIRS="$RUNDIR"
                            else
                                FINISHED_RUN_DIRS=$(printf "%s\n%s" "$FINISHED_RUN_DIRS" "$RUNDIR")
                            fi
                        fi
                    fi
                done
                if [ -z "$SUCCESSFUL_RUN_DIRS" ]; then
                    COMPLETED_RUN_COUNT=0
                else
                    COMPLETED_RUN_COUNT=$(echo "$SUCCESSFUL_RUN_DIRS" | wc -l | tr -d ' ')
                fi
                if [ -z "$FINISHED_RUN_DIRS" ]; then
                    FINISHED_RUN_COUNT=0
                else
                    FINISHED_RUN_COUNT=$(echo "$FINISHED_RUN_DIRS" | wc -l | tr -d ' ')
                fi
                NEEDS_RENUMBERING=false
                if [ "$COMPLETED_RUN_COUNT" -gt 0 ]; then
                    CURRENT_INDEX=1
                    for DIR in $SUCCESSFUL_RUN_DIRS; do
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
                DISPLAY_NAME=$(basename "$RAT_DIR")
                printf "%-20s | %10s | %10s | %10s | %s\n" "$DISPLAY_NAME" "$TOTAL_RUN_COUNT" "$COMPLETED_RUN_COUNT" "$FINISHED_RUN_COUNT" "$STATUS"
                echo "AGG_DATA:$TEMP_NAME/$DISPLAY_NAME;$TOTAL_RUN_COUNT;$COMPLETED_RUN_COUNT;$FINISHED_RUN_COUNT"
            )
        done
        echo "--------------------------------------------------------------------------------"
    )
done
EOF

    local output
    if [ "$auth_type" == "password" ]; then
        if ! command -v sshpass &> /dev/null; then
            echo "ERROR: sshpass is not installed." >&2
            return 1
        fi
        if ! sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" > /dev/null 2>&1; then
            echo "  ❌ Connection failed for $username" >&2
            return 1
        fi
        output=$(sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" 'bash -s' -- "$username" <<< "$remote_script")
    else
        if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" > /dev/null 2>&1; then
            echo "  ❌ Connection failed for $username" >&2
            return 1
        fi
        output=$(ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" 'bash -s' -- "$username" <<< "$remote_script")
    fi

    if [ $? -ne 0 ]; then
        echo "  ❌ Analysis failed for $username" >&2
        return 1
    fi
    
    echo "$output"
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
        echo "Choose authentication method for downloading:"
        echo "1) Use passwords (requires sshpass & .env file)"
        echo "2) Use SSH keys (requires key setup)"
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
        ;;
    2)
        echo "Choose authentication method for remote analysis:"
        echo "1) Use passwords (requires sshpass & .env file)"
        echo "2) Use SSH keys (requires key setup)"
        read -p "Enter your choice (1 or 2): " auth_method
        echo ""

        ALL_REMOTE_OUTPUT=""
        case $auth_method in
            1)
                echo "Using password authentication for remote analysis..."
                for username in "${USERNAMES[@]}"; do
                    password=$(get_password "$username")
                    if [ -n "$username" ] && [ -n "$password" ]; then
                        echo "================================================================================="
                        echo "Analyzing account: $username"
                        echo "================================================================================="
                        output=$(analyze_remotely "$username" "$password" "password")
                        if [ $? -eq 0 ]; then
                            ALL_REMOTE_OUTPUT+="$output\n"
                        fi
                    else
                        echo "⚠️  Skipping $username: username or password not found."
                    fi
                done
                ;;
            2)
                echo "Using SSH key authentication for remote analysis..."
                for username in "${USERNAMES[@]}"; do
                    if [ -n "$username" ]; then
                        echo "================================================================================="
                        echo "Analyzing account: $username"
                        echo "================================================================================="
                        output=$(analyze_remotely "$username" "" "key")
                        if [ $? -eq 0 ]; then
                            ALL_REMOTE_OUTPUT+="$output\n"
                        fi
                    else
                        echo "⚠️  Skipping: username not found."
                    fi
                done
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
        
        echo ""
        echo "================================================================================="
        echo "Per-Account Analysis Summary"
        echo "================================================================================="
        echo -e "$ALL_REMOTE_OUTPUT" | grep -v "AGG_DATA:"

        echo ""
        echo "================================================================================="
        echo "Combined Analysis Across All Accounts"
        echo "================================================================================="
        printf "%-30s | %10s | %10s | %10s\n" "Parameter Setting" "Submitted" "Ran" "Finished"
        printf "%-30s | %10s | %10s | %10s\n" "------------------------------" "----------" "----------" "----------"

        echo -e "$ALL_REMOTE_OUTPUT" | grep "AGG_DATA:" | sed 's/^AGG_DATA://' | \
        awk -F';' '
        {
            submitted[$1] += $2;
            ran[$1] += $3;
            finished[$1] += $4;
        }
        END {
            for (key in submitted) {
                printf "%s;%d;%d;%d\n", key, submitted[key], ran[key], finished[key];
            }
        }' | sort | \
        awk -F';' '
        {
            printf "%-30s | %10s | %10s | %10s\n", $1, $2, $3, $4;
        }'
        
        echo "================================================================================="
        echo ""
        echo "Remote analysis complete."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
