#!/usr/bin/env bash
# Download LUNA16 subsets 5-9 from Zenodo and upload to Kaggle as a private dataset.
#
# Prerequisites:
#   brew install kaggle   # or: pip install kaggle
#   kaggle config: ~/.kaggle/kaggle.json with your API key
#
# Zenodo records:
#   Part 1 (subsets 0-6 + CSVs): https://zenodo.org/records/2595813
#   Part 2 (subsets 7-9):         https://zenodo.org/records/2596479
#
# We only need subsets 5-9 since avc0706/luna16 already has 0-4.

set -euo pipefail

WORK_DIR="$HOME/luna16_upload"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "=== Downloading LUNA16 subsets 5-9 from Zenodo ==="
echo "This will download ~40 GB. Make sure you have enough disk space."
echo ""

# Part 1: subsets 5, 6 (from Zenodo record 2595813)
for i in 5 6; do
    FILE="subset${i}.zip"
    URL="https://zenodo.org/records/2595813/files/${FILE}?download=1"
    if [ ! -f "$FILE" ]; then
        echo "Downloading $FILE..."
        curl -L -o "$FILE" "$URL"
    else
        echo "$FILE already exists, skipping."
    fi
done

# Part 2: subsets 7, 8, 9 (from Zenodo record 2596479)
for i in 7 8 9; do
    FILE="subset${i}.zip"
    URL="https://zenodo.org/records/2596479/files/${FILE}?download=1"
    if [ ! -f "$FILE" ]; then
        echo "Downloading $FILE..."
        curl -L -o "$FILE" "$URL"
    else
        echo "$FILE already exists, skipping."
    fi
done

echo ""
echo "=== Extracting ==="
EXTRACT_DIR="$WORK_DIR/luna16-subsets-5-9"
mkdir -p "$EXTRACT_DIR"

for i in 5 6 7 8 9; do
    FILE="subset${i}.zip"
    echo "Extracting $FILE..."
    unzip -q -o "$FILE" -d "$EXTRACT_DIR"
done

echo ""
echo "=== Verifying ==="
MHD_COUNT=$(find "$EXTRACT_DIR" -name "*.mhd" | wc -l | tr -d ' ')
RAW_COUNT=$(find "$EXTRACT_DIR" -name "*.raw" | wc -l | tr -d ' ')
echo "Found $MHD_COUNT .mhd files and $RAW_COUNT .raw files"

echo ""
echo "=== Creating Kaggle dataset metadata ==="
cat > "$EXTRACT_DIR/dataset-metadata.json" << 'EOF'
{
    "title": "LUNA16 Subsets 5-9",
    "id": "INSERT_YOUR_KAGGLE_USERNAME/luna16-subsets-5-9",
    "licenses": [
        {"name": "CC-BY-4.0"}
    ]
}
EOF

echo ""
echo "IMPORTANT: Edit $EXTRACT_DIR/dataset-metadata.json"
echo "  Replace INSERT_YOUR_KAGGLE_USERNAME with your Kaggle username."
echo ""
echo "Then run:"
echo "  kaggle datasets create -p $EXTRACT_DIR"
echo ""
echo "This will upload the dataset to Kaggle as a private dataset."
echo "After upload completes, add it to your notebook via sidebar → Add data."
echo ""
echo "Files are in: $EXTRACT_DIR"
echo "Total size: $(du -sh "$EXTRACT_DIR" | cut -f1)"
