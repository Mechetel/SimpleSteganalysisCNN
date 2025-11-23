#!/bin/bash

# Find all directories named exactly "stego" or "cover"
folders=$(find ~/data/boss_256_0.4 -type d \( -name "stego" -o -name "cover" \))
# folders=$(find /Users/dmitryhoma/Projects/phd_dissertation/state_3/SimpleSteganalysisCNN/data/boss_256_0.4 -type d \( -name "stego" -o -name "cover" \))

for dir in $folders; do
    echo "Processing: $dir"

    # Create subfolders
    mkdir -p "$dir/train"
    mkdir -p "$dir/val"

    # Move files 1–8000.pgm → train
    for i in $(seq 1 8000); do
        file="$dir/$i.png"
        if [[ -f "$file" ]]; then
            mv "$file" "$dir/train/"
        fi
    done

    # Move files 8001–10000.pgm → val
    for i in $(seq 8001 9000); do
        file="$dir/$i.png"
        if [[ -f "$file" ]]; then
            mv "$file" "$dir/val/"
        fi
    done
done

echo "Done!"
