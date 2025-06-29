#!/bin/bash
#
# Distributes temperature folders to the corresponding Snellius accounts.
# 
# Account mapping (temp_folder → username/password) is read from a config file.
#
# Usage:
#   bash distribute_to_snellius.sh
#
# Place your account mappings in accounts.tsv:
#   TEMP_FOLDER	USERNAME	PASSWORD
#   temp_0_10	spoulain	secret123
#   temp_0_50	spoulain1	secret456
#
# DO NOT COMMIT accounts.tsv TO VERSION CONTROL.
###############################################################################

CONFIG_FILE="accounts.tsv"
SNELLIUS_HOST="snellius.surf.nl"
BASE_DIR=$(pwd)

# --- Parse account mappings ---
declare -A FOLDER_TO_USER
declare -A USER_TO_PASS

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' not found."
    echo "Please create it in the current directory."
    exit 1
fi

while IFS=$'\t' read -r temp_folder username password; do
    # Skip comments and header
    [[ "$temp_folder" =~ ^# ]] && continue
    [[ "$temp_folder" == "TEMP_FOLDER" ]] && continue
    [ -z "$temp_folder" ] && continue
    FOLDER_TO_USER["$temp_folder"]="$username"
    USER_TO_PASS["$username"]="$password"
done < "$CONFIG_FILE"

TEMP_FOLDERS=("${!FOLDER_TO_USER[@]}")

# --- Functions ---

copy_to_account() {
    local temp_folder="$1"
    local username="$2"
    local password="$3"
    local destination_path="/home/$username/"
    
    echo "========================================="
    echo "Syncing $temp_folder to $username@$SNELLIUS_HOST"
    echo "Destination: $destination_path"
    echo "========================================="
    
    if ! command -v sshpass &> /dev/null; then
        echo "ERROR: sshpass is not installed. Please install it first."
        return 1
    fi
    if ! command -v rsync &> /dev/null; then
        echo "ERROR: rsync is not installed. Please install it first."
        return 1
    fi
    
    echo "  Testing connection to $username@$SNELLIUS_HOST..."
    if sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" 2>/dev/null; then
        echo "  ✅ Connection test passed"
    else
        echo "  ❌ Connection failed - check username/password"
        return 1
    fi

    echo "  Syncing files for $temp_folder (this may take a while)..."
    if sshpass -p "$password" rsync -av --progress -e "ssh -o StrictHostKeyChecking=no" \
        "$BASE_DIR/$temp_folder" "$username@$SNELLIUS_HOST:$destination_path"; then
        echo "  ✅ Sync completed successfully"
        return 0
    else
        echo "  ❌ Sync failed"
        return 1
    fi
}

copy_to_account_with_keys() {
    local temp_folder="$1"
    local username="$2"
    local destination_path="/home/$username/"
    
    echo "========================================="
    echo "Syncing $temp_folder to $username@$SNELLIUS_HOST (using SSH keys)"
    echo "Destination: $destination_path"
    echo "========================================="
    
    if ! command -v rsync &> /dev/null; then
        echo "ERROR: rsync is not installed. Please install it first."
        return 1
    fi

    echo "  Testing connection to $username@$SNELLIUS_HOST..."
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" 2>/dev/null; then
        echo "  ✅ Connection test passed"
    else
        echo "  ❌ Connection failed - check SSH key setup"
        return 1
    fi

    echo "  Syncing files for $temp_folder (this may take a while)..."
    if rsync -av --progress -e "ssh -o StrictHostKeyChecking=no" \
        "$BASE_DIR/$temp_folder" "$username@$SNELLIUS_HOST:$destination_path"; then
        echo "  ✅ Sync completed successfully"
    else
        echo "  ❌ Sync failed"
        return 1
    fi
}

# --- Choose authentication method ---
echo "Choose authentication method:"
echo "1) Use passwords (requires sshpass)"
echo "2) Use SSH keys (more secure, requires key setup)"
read -p "Enter your choice (1 or 2): " auth_method

case $auth_method in
    1)
        echo "Using password authentication..."
        echo ""
        for temp_folder in "${TEMP_FOLDERS[@]}"; do
            username="${FOLDER_TO_USER[$temp_folder]}"
            password="${USER_TO_PASS[$username]}"
            if [ ! -d "$BASE_DIR/$temp_folder" ]; then
                echo "⚠️  Warning: $temp_folder directory not found, skipping..."
                continue
            fi
            if [ -z "$username" ] || [ -z "$password" ]; then
                echo "⚠️  Warning: No username/password found for $temp_folder, skipping..."
                continue
            fi
            copy_to_account "$temp_folder" "$username" "$password"
        done
        ;;
    2)
        echo "Using SSH key authentication..."
        echo ""
        for temp_folder in "${TEMP_FOLDERS[@]}"; do
            username="${FOLDER_TO_USER[$temp_folder]}"
            if [ ! -d "$BASE_DIR/$temp_folder" ]; then
                echo "⚠️  Warning: $temp_folder directory not found, skipping..."
                continue
            fi
            if [ -z "$username" ]; then
                echo "⚠️  Warning: No username found for $temp_folder, skipping..."
                continue
            fi
            copy_to_account_with_keys "$temp_folder" "$username"
        done
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "Distribution complete!"
echo ""
echo "Summary of distributions:"
for temp_folder in "${TEMP_FOLDERS[@]}"; do
    username="${FOLDER_TO_USER[$temp_folder]}"
    if [ -n "$username" ]; then
        echo "  $temp_folder → $username@$SNELLIUS_HOST"
    fi
done
echo ""
echo "Next steps:"
echo "1. Log into each account to verify the files were copied correctly"
echo "2. Run your simulations from the respective accounts"
echo "3. Consider setting up SSH keys for future transfers (more secure)"