"""
Contamination checker.

Checks whether eval items have leaked into the training corpus.

Three levels of checking:
1. Exact match: same text after normalization
2. Normalized match: same text after aggressive normalization
   (lowercase, strip all punctuation/whitespace)
3. Near match: MinHash Jaccard similarity above threshold

Cross-script check:
- For Devanagari eval items, we also generate a rough Latin transliteration
  and check if that appears in training. This catches the case where the
  same content exists in both scripts.

Output: contamination.jsonl with one row per affected eval item.
"""

import gzip
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.dedup import compute_exact_hash, compute_minhash, MINHASH_NUM_PERM
from src.normalize import normalize_text


# Aggressive normalization for contamination checking
def _aggressive_normalize(text: str) -> str:
    """
    Normalize text aggressively for contamination checking.
    - Lowercase (Latin only, Devanagari doesn't have case)
    - Strip all punctuation and whitespace
    - NFC normalize
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Remove all non-alphanumeric, non-Devanagari characters
    text = re.sub(r"[^\w\u0900-\u097F]", "", text)
    return text


# Simple Devanagari to Latin transliteration (phonetic approximation)
# This is NOT meant to be linguistically perfect - just good enough to
# catch obvious cross-script duplicates
_DEVANAGARI_TO_LATIN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha',
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ः': 'h', '्': '',
    '।': '.', '॥': '.',
}


def _rough_transliterate(text: str) -> str:
    """
    Rough Devanagari to Latin transliteration.
    Good enough for contamination detection, not for actual transliteration.
    """
    result = []
    for ch in text:
        if ch in _DEVANAGARI_TO_LATIN:
            result.append(_DEVANAGARI_TO_LATIN[ch])
        elif 0x0900 <= ord(ch) <= 0x097F:
            # Unknown Devanagari character - skip
            continue
        else:
            result.append(ch)
    return "".join(result)


def check_contamination(eval_path: str, clean_dir: str, out_dir: str,
                        near_threshold: float = 0.80):
    """
    Check training corpus for leaks of eval items.
    
    Returns list of contamination records.
    """
    # Load eval items
    eval_items = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    eval_items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not eval_items:
        print("WARNING: No eval items found", file=sys.stderr)
        return []

    # Build eval indices
    eval_exact_hashes: Dict[str, str] = {}      # hash -> eval_id
    eval_norm_hashes: Dict[str, str] = {}        # aggressive_norm_hash -> eval_id
    eval_translit_hashes: Dict[str, str] = {}    # transliterated_hash -> eval_id
    eval_minhashes: Dict[str, object] = {}       # eval_id -> minhash

    for item in eval_items:
        eval_id = item["id"]
        text = item["text"]
        
        # Exact hash
        exact_hash = compute_exact_hash(normalize_text(text))
        eval_exact_hashes[exact_hash] = eval_id
        
        # Aggressive normalization hash
        norm_hash = compute_exact_hash(_aggressive_normalize(text))
        eval_norm_hashes[norm_hash] = eval_id
        
        # Cross-script: transliterate Devanagari eval items
        if item.get("lang") == "hi":
            translit = _rough_transliterate(text)
            translit_hash = compute_exact_hash(_aggressive_normalize(translit))
            eval_translit_hashes[translit_hash] = eval_id
        
        # MinHash for near-duplicate detection
        mh = compute_minhash(text, MINHASH_NUM_PERM)
        eval_minhashes[eval_id] = mh

    # Scan training corpus
    contamination = []  # list of {eval_id, train_ids, method}
    contamination_map: Dict[str, List] = {}  # eval_id -> [{train_id, method}]

    clean_path = Path(clean_dir)
    for filepath in sorted(clean_path.glob("*.jsonl.gz")):
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    text = doc.get("text", "")
                    if not text:
                        continue
                    
                    train_id = f"{filepath.name}:{line_num}"
                    
                    # Check 1: Exact match
                    exact_hash = compute_exact_hash(normalize_text(text))
                    if exact_hash in eval_exact_hashes:
                        eval_id = eval_exact_hashes[exact_hash]
                        if eval_id not in contamination_map:
                            contamination_map[eval_id] = []
                        contamination_map[eval_id].append({
                            "train_id": train_id,
                            "method": "exact",
                        })
                        continue
                    
                    # Check 2: Normalized match
                    norm_hash = compute_exact_hash(_aggressive_normalize(text))
                    if norm_hash in eval_norm_hashes:
                        eval_id = eval_norm_hashes[norm_hash]
                        if eval_id not in contamination_map:
                            contamination_map[eval_id] = []
                        contamination_map[eval_id].append({
                            "train_id": train_id,
                            "method": "normalized",
                        })
                        continue
                    
                    # Check 3: Cross-script match (Latin training vs Devanagari eval)
                    if doc.get("lang") == "hi-Latn":
                        translit_hash = compute_exact_hash(
                            _aggressive_normalize(text))
                        if translit_hash in eval_translit_hashes:
                            eval_id = eval_translit_hashes[translit_hash]
                            if eval_id not in contamination_map:
                                contamination_map[eval_id] = []
                            contamination_map[eval_id].append({
                                "train_id": train_id,
                                "method": "cross_script",
                            })
                            continue
                    
                    # Check 4: Near-duplicate (MinHash)
                    train_mh = compute_minhash(text, MINHASH_NUM_PERM)
                    for eval_id, eval_mh in eval_minhashes.items():
                        sim = train_mh.jaccard(eval_mh)
                        if sim >= near_threshold:
                            if eval_id not in contamination_map:
                                contamination_map[eval_id] = []
                            contamination_map[eval_id].append({
                                "train_id": train_id,
                                "method": f"near:{sim:.3f}",
                            })
                            break
        except Exception as e:
            print(f"WARNING: Error reading {filepath}: {e}", file=sys.stderr)
            continue

    # Build output
    for eval_id, matches in contamination_map.items():
        train_ids = [m["train_id"] for m in matches]
        methods = [m["method"] for m in matches]
        contamination.append({
            "eval_id": eval_id,
            "train_ids": train_ids,
            "method": methods[0] if len(set(methods)) == 1 else ",".join(set(methods)),
            "status": "dropped",
        })

    # Write contamination report
    contam_path = os.path.join(out_dir, "contamination.jsonl")
    with open(contam_path, "w", encoding="utf-8") as f:
        for record in contamination:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nContamination check complete:")
    print(f"  Eval items checked: {len(eval_items)}")
    print(f"  Contaminated eval items: {len(contamination)}")
    print(f"  Report written to: {contam_path}")

    # Remove contaminated training documents
    if contamination:
        _remove_contaminated_training(clean_dir, contamination)

    return contamination


def _remove_contaminated_training(clean_dir: str, contamination: List[Dict]):
    """Remove contaminated documents from training files."""
    # Collect all train_ids to remove
    train_ids_to_remove = set()
    for record in contamination:
        for tid in record.get("train_ids", []):
            train_ids_to_remove.add(tid)

    if not train_ids_to_remove:
        return

    clean_path = Path(clean_dir)
    removed = 0
    for filepath in sorted(clean_path.glob("*.jsonl.gz")):
        kept = []
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    train_id = f"{filepath.name}:{line_num}"
                    if train_id in train_ids_to_remove:
                        removed += 1
                        continue
                    kept.append(line.rstrip("\n"))
        except Exception:
            continue

        # Rewrite if any removals
        if removed > 0:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")

    print(f"  Removed {removed} contaminated training documents")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m src.contamination <out_dir>")
        sys.exit(1)
    
    out_dir = sys.argv[1]
    eval_path = os.path.join(out_dir, "eval.jsonl")
    clean_dir = os.path.join(out_dir, "clean")
    check_contamination(eval_path, clean_dir, out_dir)
