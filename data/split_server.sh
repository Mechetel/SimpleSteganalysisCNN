#!/bin/bash

# Find all directories named exactly "stego" or "cover"
folders=$(find ~/data/GBRASNET/BOSSbase-1.01-div -type d \( -name "stego" -o -name "cover" \))

for dir in $folders; do
    echo "Processing: $dir"

    # Create subfolders
    mkdir -p "$dir/train"
    mkdir -p "$dir/val"

    # Move files 1–8000.pgm → train
    for i in $(seq 1 8000); do
        file="$dir/$i.pgm"
        if [[ -f "$file" ]]; then
            mv "$file" "$dir/train/"
        fi
    done

    # Move files 8001–10000.pgm → val
    for i in $(seq 8001 10000); do
        file="$dir/$i.pgm"
        if [[ -f "$file" ]]; then
            mv "$file" "$dir/val/"
        fi
    done
done

rmdir ~/data/GBRASNET/BOSSbase-1.01-div/train ~/data/GBRASNET/BOSSbase-1.01-div/val

echo "Done!"
