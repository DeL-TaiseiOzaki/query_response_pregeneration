#!/bin/bash

# Fixed paths
input_file="data/persona.json"
output_file="outputs/$(date +%Y%m%d_%H%M%S)_results.json"

# Create output directory
mkdir -p outputs

# Run the main script
echo "Starting query generation and processing..."
echo "Input file: $input_file"
python3 main.py "$input_file" "$output_file"

if [ $? -eq 0 ]; then
    echo "Processing completed. Results saved to: $output_file"
else
    echo "Error occurred during processing"
    exit 1
fi