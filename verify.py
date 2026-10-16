import gzip, json, sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

out_dir = sys.argv[1] if len(sys.argv) > 1 else "test_output"

# Check PII redaction
print("=== PII REDACTION ===")
hi_path = os.path.join(out_dir, "clean", "hi.jsonl.gz")
if os.path.exists(hi_path):
    with gzip.open(hi_path, "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if any(t in doc["text"] for t in ["<PHONE>", "<EMAIL>", "<AADHAAR>", "<PAN>"]):
                print(f"  REDACTED: {doc['text'][:150]}")
            if "कबीर" in doc["text"]:
                print(f"  PROVERB KEPT: {doc['text'][:100]}")

# Check ledger
print("\n=== LEDGER DROPS ===")
ledger_path = os.path.join(out_dir, "ledger.jsonl")
if os.path.exists(ledger_path):
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["status"] == "dropped":
                dup = entry.get("dup_of", "")
                print(f"  {entry['id']:35s} stage={entry['stage']:12s} reason={entry['reason']}" + 
                      (f" dup_of={dup}" if dup else ""))

# Check Hinglish
print("\n=== HINGLISH DOCS ===")
hl_path = os.path.join(out_dir, "clean", "hi-Latn.jsonl.gz")
if os.path.exists(hl_path):
    with gzip.open(hl_path, "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            print(f"  [{doc['lang']}] {doc['text'][:90]}...")

# Manifest
print("\n=== MANIFEST ===")
manifest_path = os.path.join(out_dir, "manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    print(f"  Counts: {json.dumps(m['counts']['per_stage'], indent=2)}")
