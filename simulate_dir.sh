#!/usr/bin/env bash

INPUT_DIR=$1
unset LD_LIBRARY_PATH
if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing it it."
    pip3 install uv
fi

echo "Syncing dependencies..."
uv sync
echo "Trying to install CUDA backend"

if uv sync --extra cuda; then
    echo "CUDA backend installed successfully."
else
    echo "CUDA backend installation failed. Proceeding without it."
fi

for png_file in "$INPUT_DIR"/*.png; do
    if [[ -f "$png_file" ]]; then
        echo "Processing file: $png_file"
        uv run python3 sim.py ${png_file} --steps 10000 --post-process-interval 20 --gif-fps 60
        echo "Finished processing $png_file"
    else
        echo "No PNG files found in the directory."
    fi
done