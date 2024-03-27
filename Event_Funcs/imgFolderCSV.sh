#!/bin/bash

# Assuming the first folder is Folder1, the second is Folder2, and the third is Folder3
Folder1="/pub/sarani/KDD/Dataset/Processed/UnifiedEventDatasetComplete/Annular"
Folder2="/pub/sarani/KDD/Event_Funcs/M255H8CSV"
Folder3="/pub/sarani/KDD/Dataset/Processed/UnifiedLSTMDataset/Annular"

# Ensure the third folder exists, create it if not
mkdir -p "$Folder3"

# Iterate over files in the second folder
for csv_file in "$Folder2"/*.csv; do
    # Extract the base filename without extension
    base_filename=$(basename "$csv_file" .csv)

    # Check if there is a corresponding image file in the first folder
    if [ -e "$Folder1/${base_filename}_image.png" ]; then
        # Copy the matching CSV file to the third folder
        cp "$csv_file" "$Folder3/"
    fi
done


Folder2="/pub/sarani/KDD/Event_Funcs/M255H154CSV"

# Ensure the third folder exists, create it if not
mkdir -p "$Folder3"

# Iterate over files in the second folder
for csv_file in "$Folder2"/*.csv; do
    # Extract the base filename without extension
    base_filename=$(basename "$csv_file" .csv)

    # Check if there is a corresponding image file in the first folder
    if [ -e "$Folder1/${base_filename}_image.png" ]; then
        # Copy the matching CSV file to the third folder
        cp "$csv_file" "$Folder3/"
    fi
done

Folder2="/pub/sarani/KDD/Event_Funcs/M255H28CSV"

# Ensure the third folder exists, create it if not
mkdir -p "$Folder3"

# Iterate over files in the second folder
for csv_file in "$Folder2"/*.csv; do
    # Extract the base filename without extension
    base_filename=$(basename "$csv_file" .csv)

    # Check if there is a corresponding image file in the first folder
    if [ -e "$Folder1/${base_filename}_image.png" ]; then
        # Copy the matching CSV file to the third folder
        cp "$csv_file" "$Folder3/"
    fi
done

Folder2="/pub/sarani/KDD/Event_Funcs/M255H48CSV"

# Ensure the third folder exists, create it if not
mkdir -p "$Folder3"

# Iterate over files in the second folder
for csv_file in "$Folder2"/*.csv; do
    # Extract the base filename without extension
    base_filename=$(basename "$csv_file" .csv)

    # Check if there is a corresponding image file in the first folder
    if [ -e "$Folder1/${base_filename}_image.png" ]; then
        # Copy the matching CSV file to the third folder
        cp "$csv_file" "$Folder3/"
    fi
done

Folder2="/pub/sarani/KDD/Event_Funcs/M255H695CSV"

# Ensure the third folder exists, create it if not
mkdir -p "$Folder3"

# Iterate over files in the second folder
for csv_file in "$Folder2"/*.csv; do
    # Extract the base filename without extension
    base_filename=$(basename "$csv_file" .csv)

    # Check if there is a corresponding image file in the first folder
    if [ -e "$Folder1/${base_filename}_image.png" ]; then
        # Copy the matching CSV file to the third folder
        cp "$csv_file" "$Folder3/"
    fi
done

