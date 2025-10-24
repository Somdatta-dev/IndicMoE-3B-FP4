#!/bin/bash

# Setup Data Directory Structure for IndicMoE-3B-FP4
# This script creates the necessary directory structure for data processing

echo "=================================================="
echo "Setting up IndicMoE-3B-FP4 Data Directory Structure"
echo "=================================================="

# Base data directory
DATA_DIR="/workspace/data"

# Create main directories
echo "Creating main directories..."
mkdir -p "$DATA_DIR/raw"
mkdir -p "$DATA_DIR/processed"
mkdir -p "$DATA_DIR/cache"

# Create language-specific directories in processed
echo "Creating language-specific directories..."
LANGUAGES=("english" "hindi" "tamil" "telugu" "bengali" "marathi" "gujarati" "kannada")

for lang in "${LANGUAGES[@]}"; do
    mkdir -p "$DATA_DIR/processed/$lang"
    echo "  ✓ Created $DATA_DIR/processed/$lang"
done

# Create phase directories in raw
echo "Creating phase directories..."
mkdir -p "$DATA_DIR/raw/phase1_pretraining"
mkdir -p "$DATA_DIR/raw/phase2_instruction"
mkdir -p "$DATA_DIR/raw/phase3_function_calling"

# Create logs directory
echo "Creating logs directory..."
mkdir -p "/workspace/logs"
mkdir -p "/workspace/runs"

# Set permissions
echo "Setting permissions..."
chmod -R 755 "$DATA_DIR"
chmod -R 755 "/workspace/logs"
chmod -R 755 "/workspace/runs"

echo ""
echo "=================================================="
echo "Directory structure created successfully!"
echo "=================================================="
echo ""
echo "Directory tree:"
tree -L 3 "$DATA_DIR" 2>/dev/null || find "$DATA_DIR" -type d | sed 's|[^/]*/| |g'

echo ""
echo "✓ Setup complete!"