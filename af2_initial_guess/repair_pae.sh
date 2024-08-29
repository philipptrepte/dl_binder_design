#!/bin/bash

# Check if a file is passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

input_file="$1"
input_dir=$(dirname "$input_file")
filename=$(basename "$input_file")
backup_file="${input_file}.backup"
temp_file="${input_file}.tmp"

# Check if the pattern exists in the input file
if grep -qE '(\d+(\.\d+)?)pae:' "$input_file"; then
    # Backup the original input file
    cp "$input_file" "$backup_file"
    # Use perl to introduce a line break between the number and "pae:"
    perl -pe 's/(\d+(\.\d+)?)pae:/\1\npae:/g' "$input_file" > "$temp_file"
    # Move the temporary file to the original file
    mv "$temp_file" "$input_file"
    echo "Error found and repairing. Original '.pae' file backed up as $backup_file"
    echo "Processed '.pae' file saved as $input_file"
else
    echo "No erros found. '.pae' file remains unchanged."
fi