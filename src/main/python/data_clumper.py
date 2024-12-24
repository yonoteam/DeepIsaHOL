# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for clumping G2TAC generated data

# #!/usr/bin/env python3

import os
import argparse
from pathlib import Path

def initialize_output_files(data_dir, mode):
    """Create or clear the output files."""
    output_dir = data_dir / "00all"
    train_path = output_dir / (mode + "_train.txt")
    validation_path = output_dir / (mode + "_validation.txt")
    test_path = output_dir / (mode + "_test.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in [train_path, validation_path, test_path]:
        with open(file_path, 'w') as f:
            f.write("")
    return train_path, validation_path, test_path

def append_to_file(file_path, line):
    """Append a single line to a file."""
    with open(file_path, 'a') as f:
        f.write(line)

def process_large_file(data_file_path, train_path, validation_path, test_path):
    """Process a large data.txt file in a memory-efficient way."""
    total_lines = sum(1 for _ in open(data_file_path, 'r'))
    train_size = int(total_lines * 0.64)
    validation_size = int(total_lines * 0.16)
    test_size = total_lines - train_size - validation_size

    print(f"Processing {data_file_path}: {total_lines} lines -> "
          f"train: {train_size}, validation: {validation_size}, test: {test_size}")

    with open(data_file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < train_size:
                append_to_file(train_path, line)
            elif i < train_size + validation_size:
                append_to_file(validation_path, line)
            else:
                append_to_file(test_path, line)

def generate_train_valid_test(data_dir, mode):
    data_dir = Path(data_dir)
    if data_dir.is_dir() and (mode in ["g2tac1", "g2tac2", "g2tac3", "g2tac4", "g2tac5"]):
        # Ensure output files are created and cleared
        train_path, validation_path, test_path = initialize_output_files(data_dir, mode)
    
        # Iterate over subdirectories and process data.txt files
        for subdir in data_dir.iterdir():
            data_file_path = subdir / (mode + ".txt")
            if data_file_path.exists():
                process_large_file(data_file_path, train_path, validation_path, test_path)
    else:
        raise ValueError(f"Not a directory: {data_dir} or invalid mode {mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, validation, and test sets.")
    parser.add_argument("data_dir", type=str, help="Path to the data directory containing subdirectories.")
    parser.add_argument("mode", type=str, choices=["g2tac1", "g2tac2", "g2tac3", "g2tac4", "g2tac5"],
                        help="Mode (e.g., g2tac1, g2tac2, g2tac3, g2tac4, g2tac5).")
    
    args = parser.parse_args()
    generate_train_valid_test(args.data_dir, args.mode)