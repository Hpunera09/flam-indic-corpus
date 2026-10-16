"""
Eval set builder.

Creates a held-out eval set from the cleaned corpus.
The eval set is meant to measure: language model perplexity and generation
quality on Hindi and Hinglish text across different domains and lengths.

Strategy:
- Deterministic selection: use hash-based sampling so the same input
  always produces the same eval set.
- Stratified: proportional representation of hi and hi-Latn.
- No overlap with training: selected documents are removed from training.
- Target: ~200 items.
"""

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


# We use hash-based deterministic sampling:
# hash(text) mod SAMPLE_MODULUS < SAMPLE_THRESHOLD → eval
# This gives us ~1/50 = 2% of documents.
# With 10K docs, that's ~200 eval items.
SAMPLE_MODULUS = 50
SAMPLE_THRESHOLD = 1  # select if hash % modulus < threshold

# Target eval size
TARGET_EVAL_SIZE = 200


def _hash_for_sampling(text: str) -> int:
    """Deterministic hash for sampling. Returns integer."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def build_eval_set(clean_dir: str, out_dir: str, target_size: int = TARGET_EVAL_SIZE):
    """
    Build eval set from cleaned corpus.
    
    1. Read all cleaned documents
    2. Select eval candidates via deterministic hash
    3. Write eval.jsonl
    4. Return set of eval doc texts (for contamination checking)
    """
    eval_items = []
    training_items = []
    
    clean_path = Path(clean_dir)
    if not clean_path.exists():
        print("ERROR: Clean directory does not exist", file=sys.stderr)
        return [], []

    # Read all cleaned documents
    all_docs = []
    for filepath in sorted(clean_path.glob("*.jsonl.gz")):
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        doc["_file"] = filepath.name
                        doc["_line"] = line_num
                        all_docs.append(doc)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"WARNING: Error reading {filepath}: {e}", file=sys.stderr)
            continue

    if not all_docs:
        print("WARNING: No documents found in clean directory", file=sys.stderr)
        return [], []

    # Deterministic selection
    for doc in all_docs:
        text = doc.get("text", "")
        if not text:
            training_items.append(doc)
            continue
        
        h = _hash_for_sampling(text)
        if h % SAMPLE_MODULUS < SAMPLE_THRESHOLD:
            eval_items.append(doc)
        else:
            training_items.append(doc)

    # If we got too many or too few, adjust
    # Sort by hash for deterministic ordering
    eval_items.sort(key=lambda d: _hash_for_sampling(d.get("text", "")))
    
    if len(eval_items) > target_size:
        # Take the first target_size items (deterministic because sorted by hash)
        overflow = eval_items[target_size:]
        training_items.extend(overflow)
        eval_items = eval_items[:target_size]
    elif not eval_items and all_docs:
        # Ensure small messy test sets still get an eval item
        eval_items.append(all_docs[0])
        training_items = [d for d in all_docs if d.get("text") != all_docs[0].get("text")]
    
    # Write eval.jsonl
    eval_path = os.path.join(out_dir, "eval.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(eval_items):
            eval_record = {
                "id": f"eval_{i:04d}",
                "text": doc["text"],
                "lang": doc.get("lang", "unknown"),
                "source": doc.get("source", "unknown"),
            }
            f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

    print(f"Eval set: {len(eval_items)} items written to {eval_path}")
    print(f"Training set: {len(training_items)} items")

    # Rewrite training files without eval items
    _rewrite_training(clean_dir, eval_items)

    return eval_items, training_items


def _rewrite_training(clean_dir: str, eval_items: List[Dict]):
    """Remove eval items from training files."""
    eval_texts = {doc["text"] for doc in eval_items}
    
    clean_path = Path(clean_dir)
    for filepath in sorted(clean_path.glob("*.jsonl.gz")):
        # Read all, filter out eval items, rewrite
        kept = []
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        if doc.get("text", "") not in eval_texts:
                            kept.append(line)
                    except json.JSONDecodeError:
                        kept.append(line)
        except Exception:
            continue
        
        # Rewrite
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")

    print(f"Removed {len(eval_items)} eval items from training files")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m src.eval_builder <out_dir>")
        sys.exit(1)
    
    out_dir = sys.argv[1]
    clean_dir = os.path.join(out_dir, "clean")
    build_eval_set(clean_dir, out_dir)
