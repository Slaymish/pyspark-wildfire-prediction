#!/bin/bash

# This script automates the download and preprocessing of the US Wildfires dataset.

# --- Configuration ---
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/rtatman/188-million-us-wildfires"
ZIP_FILE="188-million-us-wildfires.zip"
SQLITE_FILE="FPA_FOD_20170508.sqlite"
PYTHON_SCRIPT="preprocess_data.py"
OUTPUT_CSV="data/wildfire_processed_no_leakage.csv"

# --- Color Codes for Output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}--- Starting Dataset Download and Preprocessing ---${NC}"

# --- Step 0: Prerequisite Checks ---
# Check if the final processed file already exists
if [ -f "$OUTPUT_CSV" ]; then
    echo -e "${GREEN}Processed dataset '$OUTPUT_CSV' already exists. Nothing to do.${NC}"
    exit 0
fi

# Check for curl
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: 'curl' command not found. Please install it.${NC}"
    exit 1
fi

# Check for unzip
if ! command -v unzip &> /dev/null; then
    echo -e "${RED}Error: 'unzip' command not found. Please install it.${NC}"
    exit 1
fi

# Check for Python script
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Preprocessing script '$PYTHON_SCRIPT' not found in this directory.${NC}"
    exit 1
fi

# --- Step 1: Download Dataset using curl ---
echo "Step 1: Downloading dataset using curl..."
# -L follows redirects.
# -o specifies the output file name.
curl -L -o "$ZIP_FILE" "$DATASET_URL"

# Check if the download was successful by checking file size (must be > 0)
if [ ! -s "$ZIP_FILE" ]; then
    echo -e "${RED}Error: Download failed. The downloaded file is empty.${NC}"
    echo -e "${YELLOW}This can happen if Kaggle requires a login for this download. Using the Kaggle CLI might be more reliable.${NC}"
    rm -f "$ZIP_FILE" # remove empty file
    exit 1
fi
echo "Download complete."

# --- Step 2: Unzip the Dataset ---
echo "Step 2: Unzipping '$ZIP_FILE'..."
# -o overwrites files without prompting
unzip -o "$ZIP_FILE"

if [ ! -f "$SQLITE_FILE" ]; then
    echo -e "${RED}Error: Failed to unzip or '$SQLITE_FILE' not found in the archive.${NC}"
    exit 1
fi
echo "Unzip complete."

# --- Step 3: Run Python Preprocessing Script ---
echo "Step 3: Running '$PYTHON_SCRIPT' to generate the CSV..."
python3 "$PYTHON_SCRIPT"
echo "Python script finished."

# --- Step 4: Cleanup and Verification ---
echo "Step 4: Cleaning up intermediate files..."
rm "$ZIP_FILE"
rm "$SQLITE_FILE"
echo "Cleanup complete."

# Final check
if [ -f "$OUTPUT_CSV" ]; then
    echo -e "${GREEN}Success! Dataset processed and saved to '$OUTPUT_CSV'.${NC}"
else
    echo -e "${RED}Error: Process finished, but the output file '$OUTPUT_CSV' was not created.${NC}"
    exit 1
fi
