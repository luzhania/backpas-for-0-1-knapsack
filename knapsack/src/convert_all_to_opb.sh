#!/bin/bash
# Script to batch convert .in knapsack instances to .opb format

# Use current script directory as base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$SCRIPT_DIR/../instances_type14_n_5500"
OUTPUT_DIR="$BASE_DIR/backpas/dataset/14_bounded_strongly_correlated/instance"
PYTHON_SCRIPT="$SCRIPT_DIR/kp_to_opb.py"

echo "=== Starting conversion of instances from .in to OPB ==="

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

SUCCESS_COUNT=0
SKIPPED_COUNT=0
ERROR_COUNT=0

# Use find with -print0 and a while read using NUL delimiter
# to be completely robust against spaces in file/directory names.
while IFS= read -r -d '' in_file; do
    
    # 1. Extract the filename and parent directory name
    filename_in=$(basename "$in_file")
    filename_opb="${filename_in%.in}.opb"
    
    # 2. Get the name of the folder where the original instance is located (e.g. "14_bounded_strongly_correlated")
    parent_dir_name=$(basename "$(dirname "$in_file")")
    
    # 3. Build the destination path incorporating that folder
    out_dir="$OUTPUT_DIR"
    out_file="$out_dir/$filename_opb"
    
    # 4. Create the destination directory if it does not exist
    if [ ! -d "$out_dir" ]; then
        mkdir -p "$out_dir"
    fi

    # 5. Avoid overwriting (unless forced)
    if [ -f "$out_file" ]; then
        # echo "[SKIPPED] Already exists: $out_file"
        ((SKIPPED_COUNT++))
        continue
    fi
    
    # 6. Run the conversion and verify its success
    if python3 "$PYTHON_SCRIPT" "$in_file" "$out_file" > /dev/null 2>&1; then
        echo "[OK] Converted: $in_file -> $out_file"
        ((SUCCESS_COUNT++))
    else
        echo "[ERROR] Conversion failed for: $in_file"
        ((ERROR_COUNT++))
    fi

done < <(find "$INPUT_DIR" -type f -name "*.in" -print0)

echo "====================================================================="
echo "Conversion summary:"
echo "Instances converted successfully: $SUCCESS_COUNT"
echo "Instances skipped (already existed): $SKIPPED_COUNT"
echo "Instances with errors: $ERROR_COUNT"
echo "The generated instances are located in: $OUTPUT_DIR"
