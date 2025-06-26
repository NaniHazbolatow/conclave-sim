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
        echo "  ✅ Connection test passed"
    else
        echo "  ❌ Connection failed - check username/password"
        return 1
    fi
    
    # Check if remote folder exists
    echo "  Checking if remote folder exists: $remote_path"
    if sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" \
        "[ -d '$remote_path' ] && echo 'Remote folder found'" 2>/dev/null; then
        echo "  ✅ Remote folder found"
    else
        echo "  ⚠️  Remote folder not found, skipping..."
        return 1
    fi
    
    # Create local directory
    mkdir -p "$local_path"
    
    # Copy the folder from remote to local
    echo "  Downloading $temp_folder (this may take a while)..."
    if sshpass -p "$password" scp -r -o StrictHostKeyChecking=no \
        "$username@$SNELLIUS_HOST:$remote_path/*" "$local_path/" 2>/dev/null; then
        echo "  ✅ Download completed successfully"
        
        # Verify the transfer
        echo "  Verifying download..."
        remote_files=$(sshpass -p "$password" ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" \
            "find $remote_path -type f | wc -l" 2>/dev/null)
        local_files=$(find "$local_path" -type f | wc -l)
        
        echo "  Remote files: $remote_files"
        echo "  Local files: $local_files"
        
        if [ "$local_files" -gt 0 ]; then
            echo "  ✅ Download verification passed - files collected successfully"
            return 0
        else
            echo "  ⚠️  No files found locally - download may have failed"
            return 1
        fi
    else
        echo "  ❌ Download failed"
        return 1
    fi
}

# Alternative function using SSH keys (more secure)
collect_from_account_with_keys() {
    local temp_folder="$1"
    local username="$2"
    local remote_path="/home/$username/$temp_folder"
    local local_path="$COLLECTION_DIR/$temp_folder"
    
    echo "Collecting $temp_folder from $username@$SNELLIUS_HOST (using SSH keys)"
    echo "Remote: $remote_path → Local: $local_path"
    
    # Check if remote folder exists
    if ssh -o StrictHostKeyChecking=no "$username@$SNELLIUS_HOST" "[ -d '$remote_path' ]" 2>/dev/null; then
        echo "  ✅ Remote folder found"
    else
        echo "  ⚠️  Remote folder not found, skipping..."
        return 1
    fi
    
    # Create local directory
    mkdir -p "$local_path"
    
    # Copy the folder
    scp -r -o StrictHostKeyChecking=no \
        "$username@$SNELLIUS_HOST:$remote_path/*" "$local_path/"
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully collected $temp_folder from $username"
    else
        echo "  ❌ Failed to collect $temp_folder from $username"
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
        
        # Iterate through temperature folders and collect from respective accounts
        for temp_folder in "${TEMP_FOLDERS[@]}"; do
            username=$(get_username "$temp_folder")
            password=$(get_password "$username")
            
            if [ -z "$username" ] || [ -z "$password" ]; then
                echo "⚠️  Warning: No username/password found for $temp_folder, skipping..."
                continue
            fi
            
            collect_from_account "$temp_folder" "$username" "$password"
        done
        ;;
        
    2)
        echo "Using SSH key authentication..."
        echo "Make sure you have SSH keys set up for all accounts!"
        echo ""
        
        # Iterate through temperature folders and collect from respective accounts
        for temp_folder in "${TEMP_FOLDERS[@]}"; do
            username=$(get_username "$temp_folder")
            
            if [ -z "$username" ]; then
                echo "⚠️  Warning: No username found for $temp_folder, skipping..."
                continue
            fi
            
            collect_from_account_with_keys "$temp_folder" "$username"
        done
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "Collection complete!"
echo ""
echo "Summary of collections:"
for temp_folder in "${TEMP_FOLDERS[@]}"; do
    username=$(get_username "$temp_folder")
    if [ -n "$username" ]; then
        local_path="$COLLECTION_DIR/$temp_folder"
        file_count=$(find "$local_path" -type f 2>/dev/null | wc -l)
        echo "  $temp_folder ← $username@$SNELLIUS_HOST ($file_count files)"
    fi
done
echo ""
echo "All collected files are in: $COLLECTION_DIR"
echo ""
echo "Next steps:"
echo "1. Check the collected files in $COLLECTION_DIR"
echo "2. Analyze your simulation results"
echo "3. Consider backing up the collected data"
