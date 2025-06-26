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

# Function to copy folder to account using sshpass
copy_to_account() {
    local temp_folder="$1"
    local username="$2"
    local password="$3"
    local destination_path="/home/$username/"
    
    echo "========================================="
    echo "Copying $temp_folder to $username@$SNELLIUS_HOST"
    echo "Destination: $destination_path"
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
        echo "  ✅ Connection test passed"
    else
        echo "  ❌ Connection failed - check username/password"
        return 1
    fi
    
    # Create destination directory on remote server
    echo "  Creating destination directory: $destination_path"
    if sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" \
        "mkdir -p $destination_path && echo 'Directory created successfully'" 2>/dev/null; then
        echo "  ✅ Directory ready"
    else
        echo "  ❌ Could not create directory"
        return 1
    fi
    
    # Copy the folder
    echo "  Transferring $temp_folder (this may take a while)..."
    if sshpass -p "$password" scp -r -o StrictHostKeyChecking=no \
        "$BASE_DIR/$temp_folder" "$username@$SNELLIUS_HOST:$destination_path" 2>/dev/null; then
        echo "  ✅ Transfer completed successfully"
        
        # Verify the transfer
        echo "  Verifying transfer..."
        remote_files=$(sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" \
            "find $destination_path/$temp_folder -type f | wc -l" 2>/dev/null)
        local_files=$(find "$BASE_DIR/$temp_folder" -type f | wc -l)
        
        echo "  Local files: $local_files"
        echo "  Remote files: $remote_files"
        
        if [ "$remote_files" -eq "$local_files" ] && [ "$remote_files" -gt 0 ]; then
            echo "  ✅ Verification passed - all files transferred correctly"
            return 0
        else
            echo "  ⚠️  File count mismatch - transfer may be incomplete"
            return 1
        fi
    else
        echo "  ❌ Transfer failed"
        return 1
    fi
}

# Alternative function using SSH keys (more secure)
copy_to_account_with_keys() {
    local temp_folder="$1"
    local username="$2"
    local destination_path="/home/$username/simulation_runs/"
    
    echo "Copying $temp_folder to $username@$SNELLIUS_HOST:$destination_path (using SSH keys)"
    
    # Create destination directory
    ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" \
        "mkdir -p $destination_path" 2>/dev/null
    
    # Copy the folder
    scp -r -o StrictHostKeyChecking=no \
        "$BASE_DIR/$temp_folder" "$username@$SNELLIUS_HOST:$destination_path"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully copied $temp_folder to $username"
    else
        echo "  ❌ Failed to copy $temp_folder to $username"
        return 1
    fi
    
    echo ""
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
