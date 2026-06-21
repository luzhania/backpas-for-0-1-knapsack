#!/bin/bash

# Configuration
N1=1500 # train
N2=100  # valid
N3=100  # trust_region_valid
N4=100 # test

NUM_INSTANCES=$((N1 + N2 + N3 + N4))
R=3000          # Fixed range of coefficients
TYPE=14         # 14: bounded_strongly_correlated
S=1000           # Series length (allows mapping i=1..1000 to phi=0..99%)
OUTPUT_DIR="../varied_size_similar_tightness"

# Compile generator if not exists or gen2.c is newer
if [ ! -f ./gen2 ] || [ gen2.c -nt ./gen2 ]; then
    echo "Compiling gen2.c..."
    gcc -O -o gen2 gen2.c -lm
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting generation of $NUM_INSTANCES varied instances..."
echo "train: $N1, valid: $N2, trust_region_valid: $N3, test: $N4"
echo "Type=$TYPE (bounded_strongly_correlated), R=$R"
echo "Targeting n = 10000 and phi in [35, 75]"

for (( k=1; k<=NUM_INSTANCES; k++ )); do
    # Fixed n = 10000
    n=40000
    
    # Generate random i between 350 and 750
    # This corresponds to phi approximately between 35% and 75% when S=1000
    i=$(( (RANDOM % 401) + 350 ))
    
    # Calculate approximate ratio percentage for filename
    # Ratio ~ i / (S+1)
    ratio_pct=$(( (i * 100) / (S + 1) ))
    
    if (( k <= N1 )); then
        prefix="train"
    elif (( k <= N1 + N2 )); then
        prefix="valid"
    elif (( k <= N1 + N2 + N3 )); then
        prefix="trust_region_valid"
    else
        prefix="trust_region_test"
    fi

    # Create descriptive filename
    # Format: {prefix}_instance_k_type14_n{n}_phi{ratio}.in
    # Zero pad the sequence number 'k' for better sorting (e.g., 0001, 0002)
    seq_padded=$(printf "%04d" $k)
    filename="${prefix}_instance_${seq_padded}_type${TYPE}_n${n}_phi${ratio_pct}.in"
    filepath="$OUTPUT_DIR/$filename"
    
    # Progress indicator every 100 instances
    if (( k % 100 == 0 )); then
        echo "Generating instance $k of $NUM_INSTANCES (${prefix})..."
    fi
    
    # Run generator with unique seed based on sequence number k + 5001
    ./gen2 $n $R $TYPE $i $S $k > /dev/null 2>&1
    
    # Move the strictly hardcoded "test.in" output to our desired file
    if [ -f test.in ]; then
        mv test.in "$filepath"
    else
        echo "Error: failed to generate $filepath (generator might have failed)"
        exit 1
    fi
done

echo "Generation complete! $NUM_INSTANCES instances are stored in '$OUTPUT_DIR/'"
