#!/bin/bash

# Check if a folder is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Get the folder path from the argument
folder_path=$1

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Folder '$folder_path' does not exist."
    exit 1
fi

# Initialize variables for total time difference and number of files processed
total_time_difference=0
num_files=0

# Loop through all CSV files in the folder
for csv_file in "$folder_path"/*.csv; do
    # Extract filename without extension
    filename=$(basename "$csv_file" .csv)

    #echo "Processing file: $filename"

    # Read the second row of the CSV file and extract the timestamp from the first column
    #first_timestamp=$(sed -n '2s/,.*//p' "$csv_file")
    ##################################PROPHESEE###########################################
    first_timestamp=$(sed -n '2s/[^,]*,[^,]*,[^,]*,//p' "$csv_file")

    # Read the last row of the CSV file and extract the timestamp from the first column
    #last_timestamp=$(tail -n 1 "$csv_file" | cut -d',' -f1)
    ##################################PROPHESEE###########################################
    last_timestamp=$(tail -n 1 "$csv_file" | cut -d',' -f4)

    # Calculate the time difference
    time_difference=$((last_timestamp - first_timestamp))

    # Accumulate total time difference and increment the number of files processed
    total_time_difference=$((total_time_difference + time_difference))
    num_files=$((num_files + 1))

    # Display the time difference
done

# Calculate the average time difference
average_time_difference=$((total_time_difference / num_files))

# Display the average time difference
echo "Average time difference in seconds for all files: $average_time_difference"

