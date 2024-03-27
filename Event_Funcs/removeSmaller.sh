#!/bin/bash

# Specify the folder containing CSV files
csv_folder="/pub/sarani/KDD/Dataset/Processed/UnifiedLSTMDataset/ElongatedBubbly"

# Loop through CSV files in the folder
for csv_file in "$csv_folder"/*.csv; do
    # Count the number of lines in the CSV file
    line_count=$(wc -l < "$csv_file")

    # Check if the file has exactly 50,000 lines
    if [ "$line_count" -ne 50001 ]; then
        # Print a message and remove the file
        echo "Removing $csv_file as it does not have 50000 lines."
        rm "$csv_file"
    fi
done

