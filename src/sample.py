"""
Generate a random sample of the cleaned output for submission.
Uses deterministic random seed for reproducibility.
"""

import gzip
import json
import os
import random
import sys
from pathlib import Path


SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def generate_sample(clean_dir: str, out_dir: str, sample_size: int = SAMPLE_SIZE):
    """Generate a deterministic random sample of cleaned documents."""
    
    # Read all documents
    all_docs = []
    clean_path = Path(clean_dir)
    
    for filepath in sorted(clean_path.glob("*.jsonl.gz")):
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_docs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            continue

    if not all_docs:
        print("WARNING: No documents to sample from", file=sys.stderr)
        return

    # Deterministic sample
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(all_docs, min(sample_size, len(all_docs)))

    # Write sample
    sample_path = os.path.join(out_dir, "sample_1000.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for doc in sample:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Sample: {len(sample)} documents written to {sample_path}")
    
    # Print language distribution of sample
    lang_dist = {}
    for doc in sample:
        lang = doc.get("lang", "unknown")
        lang_dist[lang] = lang_dist.get(lang, 0) + 1
    print(f"Sample language distribution: {lang_dist}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m src.sample <out_dir>")
        sys.exit(1)
    
    out_dir = sys.argv[1]
    clean_dir = os.path.join(out_dir, "clean")
    generate_sample(clean_dir, out_dir)
