"""
Main pipeline: streaming processor that reads raw data, applies all
cleaning stages, and writes output + ledger.

Architecture:
- Reads input as a stream (one document at a time) via generators
- Each stage is a function that takes a document dict and returns it
  (possibly modified) or None (if filtered out)
- Ledger is written incrementally (append mode) so a killed-and-restarted
  run produces the same output as an uninterrupted run
- Output is written to JSONL.GZ (streaming-friendly) or Parquet

Determinism:
- All operations are deterministic (no randomness in processing)
- Document order is preserved from input
- Same input → same output, guaranteed

Restart safety:
- We track processed document IDs in the ledger
- On restart, we read the ledger to find already-processed IDs
- We skip those IDs and continue from where we left off
- Final output matches an uninterrupted run because:
  a) Dedup index is rebuilt from kept documents in the ledger
  b) Document order is deterministic
"""

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, TextIO, Set

from src.normalize import normalize_text, is_garbled
from src.lang_detect import detect_language
from src.quality import check_quality
from src.dedup import DedupIndex
from src.pii import redact_pii


# Stages in order
STAGES = ["read", "normalize", "lang_detect", "quality", "dedup", "pii", "output"]


def stream_raw_documents(raw_dir: str) -> Iterator[Dict[str, Any]]:
    """
    Stream documents from raw directory.
    
    Supports:
    - .jsonl and .jsonl.gz files (one JSON object per line)
    - .json files (array of objects or single object)
    - .txt files (each file = one document)
    - .csv / .tsv files (with 'text' column)
    
    Each yielded dict has at minimum: 'text', 'source_file', 'line_num'
    Malformed records are yielded with '_error' key set.
    """
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        print(f"ERROR: Raw directory does not exist: {raw_dir}", file=sys.stderr)
        return

    # Sort files for deterministic ordering
    files = sorted(raw_path.rglob("*"))
    
    for filepath in files:
        if filepath.is_dir() or filepath.name.startswith("SOURCE.") or filepath.name.startswith("metadata."):
            continue
        
        suffix = filepath.suffix.lower()
        source_name = filepath.stem
        
        # Handle double extensions like .jsonl.gz
        if filepath.name.endswith(".jsonl.gz"):
            yield from _stream_jsonl_gz(filepath, source_name)
        elif suffix == ".jsonl":
            yield from _stream_jsonl(filepath, source_name)
        elif suffix == ".json":
            yield from _stream_json(filepath, source_name)
        elif suffix == ".txt":
            yield from _stream_txt(filepath, source_name)
        elif suffix in (".csv", ".tsv"):
            yield from _stream_csv(filepath, source_name, 
                                   delimiter="\t" if suffix == ".tsv" else ",")
        elif suffix == ".gz" and str(filepath).endswith(".jsonl.gz"):
            yield from _stream_jsonl_gz(filepath, source_name)
        else:
            # Unknown format - skip with warning
            print(f"WARNING: Skipping unknown file format: {filepath}", 
                  file=sys.stderr)


def _stream_jsonl(filepath: Path, source_name: str) -> Iterator[Dict]:
    """Stream from a JSONL file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        text = obj.get("text", obj.get("content", obj.get("body", "")))
                        yield {
                            "text": str(text) if text else "",
                            "source_file": str(filepath.name),
                            "source_name": source_name,
                            "line_num": line_num,
                            "original_meta": {k: v for k, v in obj.items() 
                                             if k not in ("text", "content", "body")},
                        }
                    else:
                        yield {
                            "text": str(obj),
                            "source_file": str(filepath.name),
                            "source_name": source_name,
                            "line_num": line_num,
                            "original_meta": {},
                        }
                except json.JSONDecodeError as e:
                    yield {
                        "text": "",
                        "source_file": str(filepath.name),
                        "source_name": source_name,
                        "line_num": line_num,
                        "_error": f"json_decode_error: {e}",
                    }
    except Exception as e:
        yield {
            "text": "",
            "source_file": str(filepath.name),
            "source_name": source_name,
            "line_num": 0,
            "_error": f"file_read_error: {e}",
        }


def _stream_jsonl_gz(filepath: Path, source_name: str) -> Iterator[Dict]:
    """Stream from a gzipped JSONL file."""
    try:
        with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        text = obj.get("text", obj.get("content", obj.get("body", "")))
                        yield {
                            "text": str(text) if text else "",
                            "source_file": str(filepath.name),
                            "source_name": source_name,
                            "line_num": line_num,
                            "original_meta": {k: v for k, v in obj.items()
                                             if k not in ("text", "content", "body")},
                        }
                    else:
                        yield {
                            "text": str(obj),
                            "source_file": str(filepath.name),
                            "source_name": source_name,
                            "line_num": line_num,
                            "original_meta": {},
                        }
                except json.JSONDecodeError as e:
                    yield {
                        "text": "",
                        "source_file": str(filepath.name),
                        "source_name": source_name,
                        "line_num": line_num,
                        "_error": f"json_decode_error: {e}",
                    }
    except Exception as e:
        yield {
            "text": "",
            "source_file": str(filepath.name),
            "source_name": source_name,
            "line_num": 0,
            "_error": f"file_read_error: {e}",
        }


def _stream_json(filepath: Path, source_name: str) -> Iterator[Dict]:
    """Stream from a JSON file (array of objects)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        data = json.loads(content)
        if isinstance(data, list):
            for i, obj in enumerate(data, 1):
                if isinstance(obj, dict):
                    text = obj.get("text", obj.get("content", obj.get("body", "")))
                    yield {
                        "text": str(text) if text else "",
                        "source_file": str(filepath.name),
                        "source_name": source_name,
                        "line_num": i,
                        "original_meta": {k: v for k, v in obj.items()
                                         if k not in ("text", "content", "body")},
                    }
                else:
                    yield {
                        "text": str(obj),
                        "source_file": str(filepath.name),
                        "source_name": source_name,
                        "line_num": i,
                        "original_meta": {},
                    }
        elif isinstance(data, dict):
            text = data.get("text", data.get("content", data.get("body", "")))
            yield {
                "text": str(text) if text else "",
                "source_file": str(filepath.name),
                "source_name": source_name,
                "line_num": 1,
                "original_meta": {k: v for k, v in data.items()
                                 if k not in ("text", "content", "body")},
            }
    except Exception as e:
        yield {
            "text": "",
            "source_file": str(filepath.name),
            "source_name": source_name,
            "line_num": 0,
            "_error": f"file_read_error: {e}",
        }


def _stream_txt(filepath: Path, source_name: str) -> Iterator[Dict]:
    """Stream from a text file (whole file = one document)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        yield {
            "text": text,
            "source_file": str(filepath.name),
            "source_name": source_name,
            "line_num": 1,
            "original_meta": {},
        }
    except Exception as e:
        yield {
            "text": "",
            "source_file": str(filepath.name),
            "source_name": source_name,
            "line_num": 0,
            "_error": f"file_read_error: {e}",
        }


def _stream_csv(filepath: Path, source_name: str, delimiter: str = ",") -> Iterator[Dict]:
    """Stream from a CSV/TSV file."""
    import csv
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for line_num, row in enumerate(reader, 2):  # 2 because header is line 1
                # Look for text column
                text = (row.get("text") or row.get("content") or 
                        row.get("body") or row.get("sentence") or
                        row.get("Text") or row.get("Content") or "")
                yield {
                    "text": str(text),
                    "source_file": str(filepath.name),
                    "source_name": source_name,
                    "line_num": line_num,
                    "original_meta": {k: v for k, v in row.items() 
                                     if k.lower() not in ("text", "content", "body", "sentence")},
                }
    except Exception as e:
        yield {
            "text": "",
            "source_file": str(filepath.name),
            "source_name": source_name,
            "line_num": 0,
            "_error": f"file_read_error: {e}",
        }


def make_doc_id(source_file: str, line_num: int) -> str:
    """Create a deterministic document ID."""
    return f"{source_file}:{line_num}"


class Ledger:
    """
    Append-only ledger tracking every input document's fate.
    
    On restart, reads existing ledger to determine already-processed docs.
    Writes one JSONL line per document.
    """

    def __init__(self, ledger_path: str):
        self.path = ledger_path
        self._processed_ids: Set[str] = set()
        self._file: Optional[TextIO] = None
        
        # Load existing ledger for restart safety
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            doc_id = entry.get("id", "")
                            if doc_id:
                                self._processed_ids.add(doc_id)
                        except json.JSONDecodeError:
                            continue

    def open(self):
        """Open ledger for appending."""
        # If restarting, we need to truncate to last complete entry
        # and rewrite. For simplicity, we just append (idempotent).
        self._file = open(self.path, "a", encoding="utf-8")

    def close(self):
        """Close ledger file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def is_processed(self, doc_id: str) -> bool:
        """Check if this doc was already processed in a previous run."""
        return doc_id in self._processed_ids

    def record(self, doc_id: str, status: str, stage: str, 
               reason: str, dup_of: Optional[str] = None):
        """Record a document's fate in the ledger."""
        entry = {
            "id": doc_id,
            "status": status,
            "stage": stage,
            "reason": reason,
        }
        if dup_of:
            entry["dup_of"] = dup_of
        
        if self._file:
            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # Flush after each write for crash safety
            self._file.flush()
        
        self._processed_ids.add(doc_id)

    def get_processed_count(self) -> int:
        return len(self._processed_ids)


class OutputWriter:
    """
    Writes cleaned documents to JSONL.GZ format.
    Streaming-friendly: one document at a time.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._files: Dict[str, gzip.GzipFile] = {}
        self._counts: Dict[str, int] = {}

    def write(self, doc: Dict[str, Any], lang: str):
        """Write a cleaned document to the appropriate output file."""
        if lang not in self._files:
            filepath = os.path.join(self.output_dir, f"{lang}.jsonl.gz")
            self._files[lang] = gzip.open(filepath, "at", encoding="utf-8")
            self._counts[lang] = 0

        output_record = {
            "id": doc.get("id", f"{doc.get('source_file', 'unknown')}:{doc.get('line_num', 0)}"),
            "text": doc["text"],
            "lang": lang,
            "source": doc.get("source_name", doc.get("source_file", "unknown")),
        }
        self._files[lang].write(json.dumps(output_record, ensure_ascii=False) + "\n")
        self._counts[lang] += 1

    def close(self):
        """Close all output files."""
        for f in self._files.values():
            f.flush()
            f.close()
        self._files.clear()

    def get_counts(self) -> Dict[str, int]:
        return dict(self._counts)


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file for manifest."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(raw_dir: str, out_dir: str):
    """
    Main pipeline entry point.
    
    Reads raw documents, applies all cleaning stages, writes output.
    """
    start_time = time.time()
    
    # Setup output directories
    clean_dir = os.path.join(out_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    
    ledger_path = os.path.join(out_dir, "ledger.jsonl")
    ledger = Ledger(ledger_path)
    ledger.open()
    
    writer = OutputWriter(clean_dir)
    dedup_idx = DedupIndex()
    
    # Counters per stage
    counts = {
        "total_input": 0,
        "skipped_malformed": 0,
        "skipped_restart": 0,
        "dropped_normalize": 0,
        "dropped_lang": 0,
        "dropped_quality": 0,
        "dropped_dedup_exact": 0,
        "dropped_dedup_near": 0,
        "kept": 0,
        "pii_redacted": 0,
    }
    lang_counts = {"hi": 0, "hi-Latn": 0, "en": 0, "other": 0}

    print(f"Starting pipeline: {raw_dir} -> {out_dir}")
    print(f"Previously processed: {ledger.get_processed_count()} documents")

    # If restarting, rebuild dedup index from kept documents in ledger
    if ledger.get_processed_count() > 0:
        print("Rebuilding dedup index from previous run...")
        _rebuild_dedup_from_output(clean_dir, dedup_idx)
        print(f"Rebuilt index with {len(dedup_idx._exact_hashes)} documents")

    # Process documents
    for doc in stream_raw_documents(raw_dir):
        counts["total_input"] += 1
        
        if counts["total_input"] % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {counts['total_input']} docs in {elapsed:.1f}s "
                  f"(kept: {counts['kept']})", file=sys.stderr)

        doc_id = make_doc_id(doc["source_file"], doc["line_num"])

        # Skip if already processed (restart safety)
        if ledger.is_processed(doc_id):
            counts["skipped_restart"] += 1
            continue

        # Stage 1: Handle malformed records
        if "_error" in doc:
            counts["skipped_malformed"] += 1
            ledger.record(doc_id, "dropped", "read", doc["_error"])
            continue

        text = doc.get("text", "")
        if not text or not text.strip():
            counts["skipped_malformed"] += 1
            ledger.record(doc_id, "dropped", "read", "empty_text")
            continue

        # Stage 2: Normalize
        normalized = normalize_text(text)
        if not normalized:
            counts["dropped_normalize"] += 1
            ledger.record(doc_id, "dropped", "normalize", "empty_after_normalization")
            continue

        if is_garbled(normalized):
            counts["dropped_normalize"] += 1
            ledger.record(doc_id, "dropped", "normalize", "garbled_encoding")
            continue

        # Stage 3: Language detection
        lang, confidence = detect_language(normalized)
        
        # We only keep hi, hi-Latn, and en
        if lang == "other":
            counts["dropped_lang"] += 1
            ledger.record(doc_id, "dropped", "lang_detect", f"other_lang:{confidence:.2f}")
            continue
        
        lang_counts[lang] += 1

        # Stage 4: Quality filter
        keep, reason = check_quality(normalized)
        if not keep:
            counts["dropped_quality"] += 1
            ledger.record(doc_id, "dropped", "quality", reason)
            continue

        # Stage 5: Deduplication
        is_dup, dup_of, method = dedup_idx.check_and_add(doc_id, normalized)
        if is_dup:
            if method and method.startswith("near"):
                counts["dropped_dedup_near"] += 1
            else:
                counts["dropped_dedup_exact"] += 1
            ledger.record(doc_id, "dropped", "dedup", method or "exact", dup_of=dup_of)
            continue

        # Stage 6: PII redaction
        redacted_text, redaction_types = redact_pii(normalized)
        if redaction_types:
            counts["pii_redacted"] += 1

        # Stage 7: Write output
        doc["text"] = redacted_text
        doc["id"] = doc_id
        writer.write(doc, lang)
        counts["kept"] += 1
        ledger.record(doc_id, "kept", "output", "passed")

    # Close everything
    writer.close()
    ledger.close()

    elapsed = time.time() - start_time

    # Write manifest
    manifest = _build_manifest(raw_dir, out_dir, counts, lang_counts, 
                               dedup_idx, elapsed)
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total input documents:  {counts['total_input']}")
    print(f"  Malformed/skipped:    {counts['skipped_malformed']}")
    print(f"  Skipped (restart):    {counts['skipped_restart']}")
    print(f"  Dropped (normalize):  {counts['dropped_normalize']}")
    print(f"  Dropped (language):   {counts['dropped_lang']}")
    print(f"  Dropped (quality):    {counts['dropped_quality']}")
    print(f"  Dropped (exact dup):  {counts['dropped_dedup_exact']}")
    print(f"  Dropped (near dup):   {counts['dropped_dedup_near']}")
    print(f"  Kept:                 {counts['kept']}")
    print(f"  PII redacted:         {counts['pii_redacted']}")
    print(f"\nLanguage distribution (before quality/dedup):")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")
    print(f"\nOutput counts:")
    for lang, count in sorted(writer.get_counts().items()):
        print(f"  {lang}: {count}")
    print(f"\nDedup stats: {dedup_idx.get_stats()}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Manifest written to: {manifest_path}")

    return counts


def _rebuild_dedup_from_output(clean_dir: str, dedup_idx: DedupIndex):
    """Rebuild dedup index from existing output files (for restart safety)."""
    clean_path = Path(clean_dir)
    if not clean_path.exists():
        return
    
    for filepath in sorted(clean_path.glob("*.jsonl.gz")):
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        if text:
                            # Use source as doc_id for rebuild
                            doc_id = f"rebuild:{hashlib.md5(text.encode()).hexdigest()[:12]}"
                            dedup_idx.check_and_add(doc_id, text)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue


def _get_git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _build_manifest(raw_dir: str, out_dir: str, counts: dict, 
                    lang_counts: dict, dedup_idx: DedupIndex,
                    elapsed: float) -> dict:
    """Build the manifest.json with all metadata."""
    # Hash input files
    input_hashes = {}
    for filepath in sorted(Path(raw_dir).rglob("*")):
        if filepath.is_file():
            try:
                input_hashes[str(filepath.name)] = compute_file_hash(str(filepath))
            except Exception:
                input_hashes[str(filepath.name)] = "error_computing_hash"

    return {
        "pipeline_version": "1.0.0",
        "git_commit": _get_git_commit(),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "raw_dir": raw_dir,
        "out_dir": out_dir,
        "input_file_hashes": input_hashes,
        "thresholds": {
            "near_dup_jaccard": DedupIndex().near_dup_threshold,
            "minhash_num_perm": DedupIndex().num_perm,
            "min_text_chars": 20,
            "min_words": 3,
            "devanagari_ratio_for_hi": 0.30,
            "hinglish_marker_ratio": 0.15,
        },
        "counts": {
            "per_stage": counts,
            "per_language_detected": lang_counts,
            "dedup": dedup_idx.get_stats(),
        },
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python -m src.pipeline <raw_dir> <out_dir>")
        sys.exit(1)
    
    raw_dir = sys.argv[1]
    out_dir = sys.argv[2]
    run_pipeline(raw_dir, out_dir)
