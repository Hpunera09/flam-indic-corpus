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


def download_hindi_wiki(raw_dir: str, max_docs: int = 11000):
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
                    if count % 1000 == 0:
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


def download_hinglish(raw_dir: str, target: int = 3000):
    """Download Hinglish/Hindi-Latin data from verified open HuggingFace sources."""
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
    
    # Source 1: findnitai/english-to-hinglish (189k conversational Hinglish sentences)
    print("Downloading Hinglish data...")
    try:
        print("  Trying findnitai/english-to-hinglish...")
        ds = load_dataset("findnitai/english-to-hinglish", split="train", streaming=True)
        added = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                trans = item.get("translation", {})
                hinglish = trans.get("hi_ng", "")
                if hinglish and len(hinglish.strip()) >= 20:
                    doc = {
                        "text": hinglish.strip(),
                        "en": trans.get("en", ""),
                        "source": "findnitai/english-to-hinglish",
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                    added += 1
                    if added >= 2000:
                        break
        sources_tried.append({"name": "findnitai/english-to-hinglish", "count": added, "success": True})
        print(f"  findnitai: {added} items")
    except Exception as e:
        sources_tried.append({"name": "findnitai/english-to-hinglish", "count": 0, "success": False, "error": str(e)})
        print(f"  findnitai failed: {e}", file=sys.stderr)

    # Source 2: WillHeld/hinglish_top (task-oriented code-switched queries)
    try:
        print("  Trying WillHeld/hinglish_top...")
        ds = load_dataset("WillHeld/hinglish_top", split="train", streaming=True)
        mode = "a" if count > 0 else "w"
        added = 0
        with open(out_path, mode, encoding="utf-8") as f:
            for item in ds:
                cs_query = item.get("cs_query", "")
                if cs_query and len(cs_query.strip()) >= 15:
                    doc = {
                        "text": cs_query.strip(),
                        "en": item.get("en_query", ""),
                        "domain": item.get("domain", ""),
                        "source": "WillHeld/hinglish_top",
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                    added += 1
                    if added >= 1000:
                        break
        sources_tried.append({"name": "WillHeld/hinglish_top", "count": added, "success": True})
        print(f"  WillHeld/hinglish_top: {added} items (total: {count})")
    except Exception as e:
        sources_tried.append({"name": "WillHeld/hinglish_top", "count": 0, "success": False, "error": str(e)})
        print(f"  WillHeld/hinglish_top failed: {e}", file=sys.stderr)

    # Source 3 fallback: festvox/cmu_hinglish_dog
    if count < target:
        try:
            print("  Trying festvox/cmu_hinglish_dog...")
            ds = load_dataset("festvox/cmu_hinglish_dog", split="train", streaming=True)
            mode = "a" if count > 0 else "w"
            added = 0
            with open(out_path, mode, encoding="utf-8") as f:
                for item in ds:
                    for key in ["utterances", "text", "hinglish"]:
                        content = item.get(key)
                        if content:
                            if isinstance(content, list):
                                for utt in content:
                                    if isinstance(utt, str) and len(utt.strip()) > 15:
                                        doc = {"text": utt.strip(), "source": "festvox/cmu_hinglish_dog"}
                                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                                        count += 1
                                        added += 1
                            elif isinstance(content, str) and len(content.strip()) > 15:
                                doc = {"text": content.strip(), "source": "festvox/cmu_hinglish_dog"}
                                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                                count += 1
                                added += 1
                            break
            sources_tried.append({"name": "festvox/cmu_hinglish_dog", "count": added, "success": True})
            print(f"  festvox/cmu_hinglish_dog: {added} items (total: {count})")
        except Exception as e:
            sources_tried.append({"name": "festvox/cmu_hinglish_dog", "count": 0, "success": False, "error": str(e)})
            print(f"  festvox/cmu_hinglish_dog failed: {e}", file=sys.stderr)

    # Write metadata
    meta = {
        "name": "Hinglish / Hindi-Latin script collection",
        "sources": [s["name"] for s in sources_tried if s.get("success")],
        "license": "CC-BY-4.0 / MIT / Research Open Access",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "total_count": count,
        "details": sources_tried,
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
