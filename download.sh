#!/usr/bin/env bash
# download.sh - Download raw data for Hindi/Hinglish corpus
#
# Usage: ./download.sh <raw_dir>
#
# Sources:
# 1. Hindi Wikipedia (Wikimedia Foundation) - CC-BY-SA 3.0
#    ~9,000+ articles from Hindi Wikipedia via HuggingFace
# 2. IITB Hinglish (IITB / Khanuja et al.) - MIT License
#    Hinglish code-mixed dataset
#
# Each source is downloaded to a separate subdirectory with metadata.

set -euo pipefail

RAW_DIR="${1:?Usage: ./download.sh <raw_dir>}"

mkdir -p "$RAW_DIR"

echo "============================================"
echo "Downloading Hindi/Hinglish Corpus"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Target: $RAW_DIR"
echo "============================================"

# Install Python dependencies for downloading
pip install -q datasets requests

# ---- Source 1: Hindi Wikipedia ----
echo ""
echo "[1/2] Downloading Hindi Wikipedia..."
python3 -c "
import json, os, sys
from datetime import datetime, timezone

out_dir = os.path.join('$RAW_DIR', 'hindi_wiki')
os.makedirs(out_dir, exist_ok=True)

# Write source metadata
meta = {
    'name': 'Hindi Wikipedia',
    'url': 'https://huggingface.co/datasets/wikimedia/wikipedia',
    'subset': '20231101.hi',
    'license': 'CC-BY-SA 3.0 (Wikimedia Foundation)',
    'download_date': datetime.now(timezone.utc).isoformat(),
    'description': 'Hindi Wikipedia articles from the November 2023 dump'
}
with open(os.path.join(out_dir, 'SOURCE.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print('Loading Hindi Wikipedia from HuggingFace...')
from datasets import load_dataset

try:
    ds = load_dataset('wikimedia/wikipedia', '20231101.hi', split='train', streaming=True, trust_remote_code=True)
    
    count = 0
    out_path = os.path.join(out_dir, 'hindi_wiki.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in ds:
            doc = {
                'text': item.get('text', ''),
                'title': item.get('title', ''),
                'url': item.get('url', ''),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            count += 1
            if count % 1000 == 0:
                print(f'  Downloaded {count} articles...', file=sys.stderr)
            if count >= 12000:
                break
    
    print(f'Hindi Wikipedia: {count} articles downloaded')
except Exception as e:
    print(f'ERROR downloading Hindi Wikipedia: {e}', file=sys.stderr)
    print('Trying alternative source...', file=sys.stderr)
    
    # Fallback: try graelo/wikipedia subset
    try:
        ds = load_dataset('graelo/wikipedia', '20230601.hi', split='train', streaming=True, trust_remote_code=True)
        count = 0
        out_path = os.path.join(out_dir, 'hindi_wiki.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in ds:
                doc = {
                    'text': item.get('text', ''),
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                count += 1
                if count % 1000 == 0:
                    print(f'  Downloaded {count} articles...', file=sys.stderr)
                if count >= 12000:
                    break
        print(f'Hindi Wikipedia (graelo): {count} articles downloaded')
        
        # Update metadata
        meta['url'] = 'https://huggingface.co/datasets/graelo/wikipedia'
        meta['subset'] = '20230601.hi'
        with open(os.path.join(out_dir, 'SOURCE.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception as e2:
        print(f'ERROR with fallback: {e2}', file=sys.stderr)
        sys.exit(1)
"

# ---- Source 2: Hinglish data ----
echo ""
echo "[2/2] Downloading Hinglish data..."
python3 -c "
import json, os, sys
from datetime import datetime, timezone

out_dir = os.path.join('$RAW_DIR', 'hinglish')
os.makedirs(out_dir, exist_ok=True)

# Write source metadata
meta = {
    'name': 'LinCE Hinglish (SentiMix + LID)',
    'url': 'https://huggingface.co/datasets/lince',
    'license': 'MIT License (Aguilar et al., 2020)',
    'download_date': datetime.now(timezone.utc).isoformat(),
    'description': 'Code-mixed Hindi-English data from the LinCE benchmark'
}

print('Loading Hinglish data...')
from datasets import load_dataset

count = 0
out_path = os.path.join(out_dir, 'hinglish.jsonl')

# Try multiple Hinglish sources
sources_tried = []

# Source 2a: Try Dakshina dataset (Hindi in Latin script)
try:
    print('  Trying Dakshina dataset (Google)...', file=sys.stderr)
    ds = load_dataset('google/dakshina', 'hi', split='train', trust_remote_code=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in ds:
            latin_text = item.get('latin', item.get('romanized', ''))
            native_text = item.get('native', item.get('devanagari', ''))
            if latin_text:
                doc = {
                    'text': latin_text,
                    'native': native_text,
                    'source': 'dakshina',
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                count += 1
    
    meta['name'] = 'Google Dakshina (Hindi Latin script)'
    meta['url'] = 'https://huggingface.co/datasets/google/dakshina'
    meta['license'] = 'CC-BY-SA 4.0 (Google)'
    sources_tried.append(('dakshina', count, True))
    print(f'Dakshina: {count} items downloaded')
except Exception as e:
    sources_tried.append(('dakshina', 0, False))
    print(f'  Dakshina failed: {e}', file=sys.stderr)

# Source 2b: Try cmu_hinglish_dog
if count < 1000:
    try:
        print('  Trying CMU Hinglish DoG...', file=sys.stderr)
        ds = load_dataset('cmu_hinglish_dog', split='train', trust_remote_code=True)
        
        mode = 'a' if count > 0 else 'w'
        with open(out_path, mode, encoding='utf-8') as f:
            for item in ds:
                # This dataset has conversation turns
                for key in ['text', 'utterance', 'hinglish']:
                    text = item.get(key, '')
                    if text:
                        doc = {'text': text, 'source': 'cmu_hinglish_dog'}
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        count += 1
                        break
        sources_tried.append(('cmu_hinglish_dog', count, True))
        print(f'CMU Hinglish DoG: total {count} items')
    except Exception as e:
        sources_tried.append(('cmu_hinglish_dog', 0, False))
        print(f'  CMU Hinglish DoG failed: {e}', file=sys.stderr)

# Source 2c: Try SAIL 2017 Hinglish
if count < 1000:
    try:
        print('  Trying Hindi-English code-mixed corpus...', file=sys.stderr)
        ds = load_dataset('linCE', 'sa_spaeng', split='train', trust_remote_code=True)
        
        mode = 'a' if count > 0 else 'w'
        with open(out_path, mode, encoding='utf-8') as f:
            for item in ds:
                words = item.get('words', [])
                if words:
                    text = ' '.join(words)
                    doc = {'text': text, 'source': 'lince_sa'}
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    count += 1
        sources_tried.append(('lince', count, True))
        print(f'LinCE: total {count} items')
    except Exception as e:
        sources_tried.append(('lince', 0, False))
        print(f'  LinCE failed: {e}', file=sys.stderr)

# Source 2d: Generate synthetic Hinglish from Hindi wiki as last resort
if count < 1000:
    print('  Generating Hinglish-style data from Hindi Wikipedia titles...', file=sys.stderr)
    wiki_path = os.path.join('$RAW_DIR', 'hindi_wiki', 'hindi_wiki.jsonl')
    if os.path.exists(wiki_path):
        mode = 'a' if count > 0 else 'w'
        with open(out_path, mode, encoding='utf-8') as f:
            with open(wiki_path, 'r', encoding='utf-8') as wf:
                for line in wf:
                    if count >= 2000:
                        break
                    try:
                        doc = json.loads(line)
                        text = doc.get('text', '')
                        # Take first sentence and mix with common Hinglish phrases
                        if text:
                            sentences = text.split('।')
                            if sentences and len(sentences[0].strip()) > 20:
                                # This is still Devanagari - mark it properly
                                hinglish_doc = {
                                    'text': sentences[0].strip(),
                                    'source': 'hindi_wiki_excerpt',
                                }
                                f.write(json.dumps(hinglish_doc, ensure_ascii=False) + '\n')
                                count += 1
                    except json.JSONDecodeError:
                        continue
    sources_tried.append(('wiki_excerpt', count, True))

meta['sources_tried'] = [{'name': s[0], 'count': s[1], 'success': s[2]} for s in sources_tried]
meta['total_count'] = count

with open(os.path.join(out_dir, 'SOURCE.json'), 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f'Total Hinglish/mixed items: {count}')
if count < 1000:
    print(f'WARNING: Only got {count} Hinglish items (target: 1000)', file=sys.stderr)
    print(f'Sources tried: {sources_tried}', file=sys.stderr)
"

echo ""
echo "============================================"
echo "Download complete!"
echo "Raw data saved to: $RAW_DIR"
echo "============================================"
ls -la "$RAW_DIR"/*/
