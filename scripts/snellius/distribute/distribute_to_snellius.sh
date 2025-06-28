#!/bin/bash

# Script to distribute temperature folders to different Snellius accounts
# Usage: bash distribute_to_snellius.sh

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

# Base directory containing temperature folders
BASE_DIR=$(pwd)
SNELLIUS_HOST="snellius.surf.nl"

echo "Starting distribution of temperature folders to Snellius accounts..."
echo "Base directory: $BASE_DIR"
echo ""

# Function to copy folder to account using sshpass and rsync
copy_to_account() {
    local temp_folder="$1"
    local username="$2"
    local password="$3"
    local destination_path="/home/$username/"
    
    echo "========================================="
    echo "Syncing $temp_folder to $username@$SNELLIUS_HOST"
    echo "Destination: $destination_path"
    echo "========================================="
    
    # Check if sshpass and rsync are installed
    if ! command -v sshpass &> /dev/null; then
        echo "ERROR: sshpass is not installed. Please install it first."
        return 1
    fi
    if ! command -v rsync &> /dev/null; then
        echo "ERROR: rsync is not installed. Please install it first."
        return 1
    fi
    
    # Test connection first
    echo "  Testing connection to $username@$SNELLIUS_HOST..."
    if sshpass -p "$password" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" 2>/dev/null; then
        echo "  ✅ Connection test passed"
    else
        echo "  ❌ Connection failed - check username/password"
        return 1
    fi
    
    # Sync the folder using rsync
    echo "  Syncing files for $temp_folder (this may take a while)..."
    # Using -a (archive), -v (verbose), --progress.
    # This efficiently syncs the directory, overwriting remote files if the local ones are newer.
    if sshpass -p "$password" rsync -av --progress -e "ssh -o StrictHostKeyChecking=no" \
        "$BASE_DIR/$temp_folder" "$username@$SNELLIUS_HOST:$destination_path"; then
        echo "  ✅ Sync completed successfully"
        return 0
    else
        echo "  ❌ Sync failed"
        return 1
    fi
}

# Alternative function using SSH keys (more secure)
copy_to_account_with_keys() {
    local temp_folder="$1"
    local username="$2"
    local destination_path="/home/$username/" # Corrected path to be consistent
    
    echo "========================================="
    echo "Syncing $temp_folder to $username@$SNELLIUS_HOST (using SSH keys)"
    echo "Destination: $destination_path"
    echo "========================================="

    if ! command -v rsync &> /dev/null; then
        echo "ERROR: rsync is not installed. Please install it first."
        return 1
    fi

    # Test connection
    echo "  Testing connection to $username@$SNELLIUS_HOST..."
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$username@$SNELLIUS_HOST" "echo 'Connection test successful'" 2>/dev/null; then
        echo "  ✅ Connection test passed"
    else
        echo "  ❌ Connection failed - check SSH key setup"
        return 1
    fi
    
    # Sync the folder using rsync
    echo "  Syncing files for $temp_folder (this may take a while)..."
    if rsync -av --progress -e "ssh -o StrictHostKeyChecking=no" \
        "$BASE_DIR/$temp_folder" "$username@$SNELLIUS_HOST:$destination_path"; then
        echo "  ✅ Sync completed successfully"
    else
        echo "  ❌ Sync failed"
        return 1
    fi
}

# Check what method to use
echo "Choose authentication method:"
echo "1) Use passwords (requires sshpass)"
echo "2) Use SSH keys (more secure, but requires key setup)"
read -p "Enter your choice (1 or 2): " auth_method

case $auth_method in
    1)
        echo "Using password authentication..."
        echo ""
        
        # Iterate through temperature folders and copy to respective accounts
        for temp_folder in "${TEMP_FOLDERS[@]}"; do
            username=$(get_username "$temp_folder")
            password=$(get_password "$username")
            
            # Check if folder exists
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
        echo "Make sure you have SSH keys set up for all accounts!"
        echo ""
        
        # Iterate through temperature folders and copy to respective accounts
        for temp_folder in "${TEMP_FOLDERS[@]}"; do
            username=$(get_username "$temp_folder")
            
            # Check if folder exists
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
    username=$(get_username "$temp_folder")
    if [ -n "$username" ]; then
        echo "  $temp_folder → $username@$SNELLIUS_HOST"
    fi
done
echo ""
echo "Next steps:"
echo "1. Log into each account to verify the files were copied correctly"
echo "2. Run your simulations from the respective accounts"
echo "3. Consider setting up SSH keys for future transfers (more secure)"
