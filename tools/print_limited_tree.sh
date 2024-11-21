#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/folder"
    exit 1
fi

folder="$1"

find "$folder" -type l -exec sh -c '
    echo "Link: {} -> $(readlink -f {})"
    readlink -f {} | xargs -I{} tree -L 2 {} | head -n 15
' \;
