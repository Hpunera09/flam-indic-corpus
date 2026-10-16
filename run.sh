#!/usr/bin/env bash
# run.sh - Main pipeline script (works offline)
#
# Usage: ./run.sh <raw_dir> <out_dir>
#
# Steps:
# 1. Install dependencies (offline-capable with pip cache)
# 2. Run cleaning pipeline
# 3. Build eval set
# 4. Run contamination check
# 5. Generate sample

set -euo pipefail

RAW_DIR="${1:?Usage: ./run.sh <raw_dir> <out_dir>}"
OUT_DIR="${2:?Usage: ./run.sh <raw_dir> <out_dir>}"

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "Flam Corpus Pipeline"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Input:  $RAW_DIR"
echo "Output: $OUT_DIR"
echo "============================================"

# Step 0: Install dependencies
echo ""
echo "[Step 0] Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || {
    echo "WARNING: pip install failed. Assuming dependencies are already installed."
}

# Step 1: Run cleaning pipeline
echo ""
echo "[Step 1] Running cleaning pipeline..."
PYTHONPATH="$SCRIPT_DIR" python3 -m src.pipeline "$RAW_DIR" "$OUT_DIR"

# Step 2: Build eval set
echo ""
echo "[Step 2] Building eval set..."
PYTHONPATH="$SCRIPT_DIR" python3 -m src.eval_builder "$OUT_DIR"

# Step 3: Run contamination check
echo ""
echo "[Step 3] Checking for contamination..."
PYTHONPATH="$SCRIPT_DIR" python3 -m src.contamination "$OUT_DIR"

# Step 4: Generate sample
echo ""
echo "[Step 4] Generating 1000-row sample..."
PYTHONPATH="$SCRIPT_DIR" python3 -m src.sample "$OUT_DIR"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo ""
echo "Output:"
echo "  Clean corpus:     $OUT_DIR/clean/"
echo "  Ledger:           $OUT_DIR/ledger.jsonl"
echo "  Eval set:         $OUT_DIR/eval.jsonl"
echo "  Contamination:    $OUT_DIR/contamination.jsonl"
echo "  Manifest:         $OUT_DIR/manifest.json"
echo "  Sample:           $OUT_DIR/sample_1000.jsonl"
echo ""

# Show file sizes
echo "File sizes:"
if command -v du &> /dev/null; then
    du -sh "$OUT_DIR"/* 2>/dev/null || true
else
    ls -la "$OUT_DIR"/ 2>/dev/null || true
fi
