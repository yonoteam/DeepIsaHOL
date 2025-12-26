#!/bin/bash

# --- install_py_env.sh ---
# Script to automatically create and activate the Python environment
# Requires Conda or Mamba to be installed. Mamba is recommended for speed.
# If you agree that the operations below are safe, 
# make this script executable: chmod +x install_py_env.sh
# and run it: ./install_py_env.sh

ENV_NAME="IsaGeneration"
YML_FILE="./src/main/python/isa_generation.yml"
MAMBA_CMD="mamba" # Default to mamba

# 1. Check for Mamba/Conda
if command -v mamba &> /dev/null; then
    MAMBA_CMD="mamba"
elif command -v conda &> /dev/null; then
    MAMBA_CMD="conda"
else
    echo "ERROR: Neither 'mamba' nor 'conda' command found."
    echo "Please install Miniconda or Mambaforge and try again."
    exit 1
fi

echo "--- Using $MAMBA_CMD to manage environment ---"

# 2. Check for environment.yml
if [ ! -f "$YML_FILE" ]; then
    echo "ERROR: Environment file '$YML_FILE' not found."
    echo "Please ensure '$YML_FILE' is in the current directory."
    exit 1
fi

# 3. Create or Update Environment
echo "--- Creating/Updating environment '$ENV_NAME' ---"
# The 'mamba env create' command handles both creation and updates 
# if the environment already exists.
# Using -f for file and -n for environment name (redundant with YML, but safe)
$MAMBA_CMD env create -f "$YML_FILE" -n "$ENV_NAME"

# Check if the environment creation/update was successful
if [ $? -eq 0 ]; then
    echo "--- Environment '$ENV_NAME' created/updated successfully ---"
    
    # --- 4. Activation Instructions ---
    # Cannot activate the environment from within this script in a way 
    # that affects the user's interactive shell, so we provide instructions.
    echo ""
    echo "============================================================"
    echo "SUCCESS! To start working, please run the following command:"
    echo "  $MAMBA_CMD activate $ENV_NAME"
    echo "============================================================"
else
    echo "--- ERROR: Failed to create/update environment '$ENV_NAME' ---"
    exit 1
fi