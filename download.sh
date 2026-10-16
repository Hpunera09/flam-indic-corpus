#!/usr/bin/env bash
# download.sh - Download raw data for Hindi/Hinglish corpus
#
# Usage: ./download.sh <raw_dir>
#
# Sources:
# 1. Hindi Wikipedia (Wikimedia Foundation) - CC-BY-SA 3.0
#    ~10,000+ articles from Hindi Wikipedia via HuggingFace
# 2. Hinglish / Hindi-Latin Script Collection:
#    - findnitai/english-to-hinglish (Conversational Hinglish)
#    - WillHeld/hinglish_top (Task-oriented code-switched queries)
#    - festvox/cmu_hinglish_dog (Dialogues)
#
# Each source is downloaded to a separate subdirectory with SOURCE.json metadata.

set -euo pipefail

RAW_DIR="${1:?Usage: ./download.sh <raw_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$RAW_DIR"

echo "============================================"
echo "Downloading Hindi/Hinglish Corpus"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Target: $RAW_DIR"
echo "============================================"

# Choose python command
if command -v python3 &>/dev/null; then
    PY="python3"
else
    PY="python"
fi

# Install dependencies if needed
$PY -m pip install -q datasets requests

# Run download
$PY "$SCRIPT_DIR/download_data.py" "$RAW_DIR"

echo ""
echo "============================================"
echo "Download complete!"
echo "Raw data saved to: $RAW_DIR"
echo "============================================"
ls -la "$RAW_DIR"/*/
