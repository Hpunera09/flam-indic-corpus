"""
download_data.py - Download raw Hindi and Hinglish corpus data.

Usage: python download_data.py <raw_dir>

Sources:
1. Hindi Wikipedia via HuggingFace (CC-BY-SA 3.0)
2. Google Dakshina Hindi Latin script (CC-BY-SA 4.0)

This script downloads to <raw_dir>/ with subdirectories per source,
each containing a SOURCE.json metadata file.
"""

import json
import os
import sys
from datetime import datetime, timezone


def download_hindi_wiki(raw_dir: str, max_docs: int = 12000):
    """Download Hindi Wikipedia articles."""
    out_dir = os.path.join(raw_dir, "hindi_wiki")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hindi_wiki.jsonl")
    
    # Check if already downloaded
    if os.path.exists(out_path):
        line_count = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
        if line_count >= max_docs:
            print(f"Hindi Wikipedia already downloaded ({line_count} articles). Skipping.")
            return line_count
    
    print("Downloading Hindi Wikipedia from HuggingFace...")
    from datasets import load_dataset
    
    # Try primary source
    sources = [
        ("wikimedia/wikipedia", "20231101.hi"),
        ("graelo/wikipedia", "20230601.hi"),
    ]
    
    for ds_name, ds_config in sources:
        try:
            print(f"  Trying {ds_name}/{ds_config}...")
            ds = load_dataset(ds_name, ds_config, split="train", 
                            streaming=True, trust_remote_code=True)
            
            count = 0
            with open(out_path, "w", encoding="utf-8") as f:
                for item in ds:
                    text = item.get("text", "")
                    if not text or len(text.strip()) < 50:
                        continue
                    doc = {
                        "text": text,
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 2000 == 0:
                        print(f"    {count} articles...", file=sys.stderr)
                    if count >= max_docs:
                        break
            
            # Write metadata
            meta = {
                "name": "Hindi Wikipedia",
                "url": f"https://huggingface.co/datasets/{ds_name}",
                "subset": ds_config,
                "license": "CC-BY-SA 3.0 (Wikimedia Foundation)",
                "download_date": datetime.now(timezone.utc).isoformat(),
                "document_count": count,
            }
            with open(os.path.join(out_dir, "SOURCE.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            
            print(f"  Hindi Wikipedia: {count} articles downloaded")
            return count
            
        except Exception as e:
            print(f"  Failed with {ds_name}: {e}", file=sys.stderr)
            continue
    
    print("ERROR: Could not download Hindi Wikipedia from any source", file=sys.stderr)
    return 0


def download_hinglish(raw_dir: str, target: int = 2000):
    """Download Hinglish/Hindi-Latin data."""
    out_dir = os.path.join(raw_dir, "hinglish")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hinglish.jsonl")
    
    # Check if already downloaded
    if os.path.exists(out_path):
        line_count = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
        if line_count >= target:
            print(f"Hinglish already downloaded ({line_count} items). Skipping.")
            return line_count
    
    from datasets import load_dataset
    
    count = 0
    sources_tried = []
    
    # Source 1: Google Dakshina - Hindi in Latin script
    print("Downloading Hinglish data...")
    try:
        print("  Trying Google Dakshina...")
        ds = load_dataset("google/dakshina", "hi", trust_remote_code=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            for split_name in ["train", "validation", "test"]:
                if split_name in ds:
                    for item in ds[split_name]:
                        # Dakshina has 'latin' and 'native' fields
                        latin = item.get("latin", item.get("romanized", ""))
                        if latin and len(latin.strip()) > 10:
                            doc = {
                                "text": latin,
                                "native": item.get("native", item.get("devanagari", "")),
                                "source": "dakshina",
                            }
                            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                            count += 1
        
        sources_tried.append({"name": "dakshina", "count": count, "success": True})
        print(f"  Dakshina: {count} items")
    except Exception as e:
        sources_tried.append({"name": "dakshina", "count": 0, "success": False, "error": str(e)})
        print(f"  Dakshina failed: {e}", file=sys.stderr)

    # Source 2: Try IITB Hinglish or similar
    if count < target:
        try:
            print("  Trying cmu_hinglish_dog...")
            ds = load_dataset("cmu_hinglish_dog", split="train", trust_remote_code=True)
            
            mode = "a" if count > 0 else "w"
            added = 0
            with open(out_path, mode, encoding="utf-8") as f:
                for item in ds:
                    for key in ["utterances", "text", "hinglish"]:
                        content = item.get(key)
                        if content:
                            if isinstance(content, list):
                                for utt in content:
                                    if isinstance(utt, str) and len(utt.strip()) > 10:
                                        doc = {"text": utt, "source": "cmu_hinglish_dog"}
                                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                                        count += 1
                                        added += 1
                            elif isinstance(content, str) and len(content.strip()) > 10:
                                doc = {"text": content, "source": "cmu_hinglish_dog"}
                                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                                count += 1
                                added += 1
                            break
            
            sources_tried.append({"name": "cmu_hinglish_dog", "count": added, "success": True})
            print(f"  cmu_hinglish_dog: {added} items (total: {count})")
        except Exception as e:
            sources_tried.append({"name": "cmu_hinglish_dog", "count": 0, "success": False, "error": str(e)})
            print(f"  cmu_hinglish_dog failed: {e}", file=sys.stderr)

    # Source 3: Try SemEval Hinglish sentiment dataset
    if count < target:
        try:
            print("  Trying SemEval Hinglish sentiment...")
            ds = load_dataset("Aman-J/Hinglish_Sentiment_Analysis", 
                            split="train", trust_remote_code=True)
            
            mode = "a" if count > 0 else "w"
            added = 0
            with open(out_path, mode, encoding="utf-8") as f:
                for item in ds:
                    text = item.get("text", item.get("sentence", ""))
                    if text and len(text.strip()) > 10:
                        doc = {"text": text, "source": "hinglish_sentiment"}
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        count += 1
                        added += 1
            
            sources_tried.append({"name": "hinglish_sentiment", "count": added, "success": True})
            print(f"  Hinglish sentiment: {added} items (total: {count})")
        except Exception as e:
            sources_tried.append({"name": "hinglish_sentiment", "count": 0, "success": False, "error": str(e)})
            print(f"  Hinglish sentiment failed: {e}", file=sys.stderr)

    # Write metadata
    meta = {
        "name": "Hinglish / Hindi-Latin script collection",
        "sources_tried": sources_tried,
        "license": "CC-BY-SA 4.0 (Dakshina), MIT (others)",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "total_count": count,
    }
    with open(os.path.join(out_dir, "SOURCE.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if count < 1000:
        print(f"\nWARNING: Only got {count} Hinglish items (target: 1000)")
        print(f"Sources tried: {json.dumps(sources_tried, indent=2)}")
    
    return count


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_data.py <raw_dir>")
        sys.exit(1)
    
    raw_dir = sys.argv[1]
    os.makedirs(raw_dir, exist_ok=True)
    
    print("=" * 60)
    print("Hindi/Hinglish Corpus Downloader")
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print(f"Target: {raw_dir}")
    print("=" * 60)
    
    hindi_count = download_hindi_wiki(raw_dir)
    hinglish_count = download_hinglish(raw_dir)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"  Hindi Wikipedia: {hindi_count} documents")
    print(f"  Hinglish: {hinglish_count} documents")
    print(f"  Total: {hindi_count + hinglish_count} documents")
    print("=" * 60)


if __name__ == "__main__":
    main()
