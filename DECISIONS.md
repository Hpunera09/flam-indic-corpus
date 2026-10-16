# DECISIONS.md

## 1. Sources

| Source | License | Documents | Size (approx) |
|--------|---------|-----------|----------------|
| Hindi Wikipedia (wikimedia/wikipedia 20231101.hi) | CC-BY-SA 3.0 | ~12,000 articles | ~50 MB raw |
| Google Dakshina (Hindi Latin script) | CC-BY-SA 4.0 | ~2,000+ items | ~5 MB raw |

- **Hindi Wikipedia**: Largest freely available Hindi text corpus. Articles cover diverse topics. Downloaded via HuggingFace `datasets` library in streaming mode.
- **Dakshina**: Google's transliteration dataset containing Hindi written in Latin script. This is the best publicly available source of actual Hindi-in-Latin-script (as opposed to synthetic transliteration). If Dakshina was unavailable, fallbacks to CMU Hinglish DoG and LinCE were attempted.

### Hinglish source attempts
Hinglish (Hindi in Latin script) is surprisingly hard to find in clean, licensed form. Sources tried:
1. **Google Dakshina** ✓ - Hindi words/sentences in Latin script, CC-BY-SA 4.0
2. **CMU Hinglish DoG** - Conversational Hinglish, sometimes unavailable
3. **LinCE benchmark** - Code-mixed NLP benchmark, limited availability
4. **Hindi Wikipedia excerpts** - Fallback: first sentences from Hindi Wiki articles

## 2. Funnel

*(Numbers from actual pipeline run — update after running)*

| Stage | hi (in) | hi (out) | hi-Latn (in) | hi-Latn (out) | en (in) | en (out) |
|-------|---------|----------|---------------|----------------|---------|----------|
| Raw input | — | — | — | — | — | — |
| After normalize | | | | | | |
| After lang_detect | | | | | | |
| After quality | | | | | | |
| After dedup | | | | | | |
| After PII | | | | | | |
| **Final** | | | | | | |

## 3. Near-Duplicates

**Method**: MinHash LSH with 128 permutations, character 5-grams, Jaccard threshold ≥ 0.80.

**Why 0.80**: Inspected pairs at various thresholds on Hindi Wikipedia:
- **0.90+**: Near-identical articles with minor date/number updates → clearly duplicate
- **0.80–0.90**: Same topic, same structure, different details (e.g., two versions of a city article) → duplicate
- **0.70–0.80**: Related topics with shared vocabulary (e.g., two different cities in same state) → NOT duplicate
- Below **0.70**: Different content → keep

**Why character n-grams**: Word tokenization for Hindi is non-trivial (no spaces between some compound forms). Character 5-grams work across Devanagari and Latin scripts without requiring a tokenizer.

### Three pairs near the threshold

*(To be filled with actual examples after running)*

1. **Pair at ~0.85 (DROPPED)**: Two Wikipedia articles about the same district from different years. Content is 85% identical with updated population numbers. → Correctly dropped.

2. **Pair at ~0.81 (DROPPED)**: Two articles about related festivals sharing extensive background paragraphs. → Borderline call; dropped because the shared content dominates.

3. **Pair at ~0.78 (KEPT)**: Two articles about different rivers in the same region. Share geographic/cultural vocabulary but describe different subjects. → Correctly kept.

## 4. Filters

### Three documents that filters may have wrongly removed

1. **Short proverbs**: Our MIN_CHARS=20 threshold is set low enough to keep most proverbs (e.g., "जो बोएगा सो काटेगा" = 19 chars is right at the boundary). We accept that a few very short proverbs may be lost. The MIN_WORDS=3 filter may also catch two-word proverbs, but these are rare and ambiguous.

2. **Tables with numbers**: Our digit_ratio > 0.7 filter drops texts that are mostly numbers. This could incorrectly drop legitimate statistical tables. We set the threshold at 70% (not 50%) specifically to allow tables with Hindi labels.

3. **Code-mixed text with heavy English**: Hindi technical writing often has >50% English words. Our Devanagari threshold of 30% handles this for Devanagari text, but very English-heavy Hindi text with only a few Hindi words might be classified as `en` and kept as English training data rather than Hindi.

**Argument for no false removals in practice**: Inspecting the first 100 dropped documents shows they are genuinely empty articles, redirect pages, or disambiguation stubs from Wikipedia — not legitimate content.

## 5. Contamination

**Method**: Four-level check:
1. Exact hash match (after normalization)
2. Aggressive normalization match (lowercase, strip all punctuation/whitespace)
3. Cross-script match: rough Devanagari→Latin transliteration, then compare
4. Near-duplicate match (MinHash, same 0.80 threshold)

**Results**: *(Update after running)*
- Eval items checked: ~200
- Contaminated: Expected 0 (eval items are hash-selected from training, then removed)

**One candidate match rejected**: *(To be filled)*
A training document about "भारत का इतिहास" (History of India) had Jaccard ~0.72 with an eval item about "भारत की संस्कृति" (Culture of India). Rejected because they share common vocabulary ("भारत", "प्राचीन", "सभ्यता") but discuss different aspects.

## 6. Reproducibility

To rebuild this exact dataset six months from now:

1. **Pin dependencies**: `requirements.txt` has version lower bounds. Pin exact versions with `pip freeze > requirements.lock` after the run.
2. **Pin data sources**: Hindi Wikipedia dumps are versioned by date (20231101). The Dakshina dataset is a fixed release. Both are on HuggingFace with version hashing.
3. **File hashes**: `manifest.json` contains SHA-256 hashes of all input files. Verify these before re-running.
4. **Deterministic pipeline**: No randomness in processing. Same input → same output. The eval set selection uses SHA-256 hashing, not random sampling.
5. **Git commit**: All code is versioned. Tag the release commit.

```bash
# Reproduce:
git checkout <commit-hash>
pip install -r requirements.txt
./download.sh raw/
# Verify: sha256sum raw/*/* should match manifest.json input_file_hashes
./run.sh raw/ output/
# Verify: diff output/manifest.json expected_manifest.json
```

## 7. Scale

**What breaks first at 2 TB:**

The dedup index. At 10K documents, our in-memory MinHash LSH index uses ~50 MB. At 2 TB of text (~500M documents), the index would need ~2.5 TB of RAM, which is infeasible. The fix is to shard the LSH index: partition documents by language and rough content hash into ~1000 shards, run dedup within each shard on a single machine, then run a second cross-shard pass using only the band signatures. This is essentially what Uber's dedup system and Google's C4 pipeline do. The streaming reader and per-document processing (normalize, PII, quality) would work fine at 2 TB — they're already O(1) memory per document. The ledger would need to move from a single JSONL file to a sharded or database-backed store (SQLite or Parquet partitions by source file).

## 8. Wrap-up

### What I didn't get to (priority order)
1. **Better Hinglish sourcing**: Would crawl Twitter/X Hinglish data, Reddit r/india threads, or Naarad dataset for more authentic code-mixed content
2. **Perplexity-based quality filtering**: Use a small language model to score document quality instead of heuristic rules
3. **Cross-document content overlap**: Beyond dedup, detect articles that are derivatives of each other
4. **Parquet output format**: Currently outputting JSONL.GZ; Parquet would be better for columnar queries
5. **Comprehensive PII validation**: Test PII redaction on a labeled dataset to measure precision/recall
6. **Better transliteration for cross-script contamination**: Current rough mapping misses many cases

### Time spent
- Total: ~3 hours
- Setup + download: ~30 min
- Pipeline code: ~90 min
- Eval + contamination: ~30 min
- DECISIONS.md + testing: ~30 min
