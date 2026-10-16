# DECISIONS.md

## 1. Sources

| Source | License | Documents | Size (approx) |
|--------|---------|-----------|----------------|
| Hindi Wikipedia (`wikimedia/wikipedia` `20231101.hi`) | CC-BY-SA 3.0 | 11,000 articles | ~70.5 MB raw |
| `findnitai/english-to-hinglish` (Conversational Hinglish) | CC-BY-4.0 | 2,000 sentences | ~250 KB raw |
| `WillHeld/hinglish_top` (Task-oriented Hinglish queries) | Research Open Access | 1,000 queries | ~150 KB raw |
| **Total Raw Collected** | Open / Public | **14,000 documents** | **~71 MB raw** |

- **Hindi Wikipedia**: The standard primary public corpus for Devanagari Hindi. Contains diverse topics including history, geography, science, biographies, and culture. Streamed directly from HuggingFace (`wikimedia/wikipedia`, dump `20231101.hi`).
- **Hinglish / Hindi-Latin Script**: Real-world Hinglish in Latin script is notoriously scarce in clean, licensed form. We sourced 3,000 genuine code-mixed documents from two verified public datasets:
  1. `findnitai/english-to-hinglish`: 2,000 authentic conversational Hinglish sentences (e.g. *"film ka kya naam hai"*, *"namaste, sada hua tomatoes score mahaan hai..."*).
  2. `WillHeld/hinglish_top`: 1,000 natural code-switched queries and commands (e.g. *"9 : 30 am ko Sunday Brunch ke liye ek naya weekly reminder add karen"*).

### Hinglish source attempts & findings
1. **Google Dakshina**: Legacy repository is not formatted as an HF dataset script on current HuggingFace Hub.
2. **findnitai/english-to-hinglish** ✓: 189k conversational Hinglish sentences. High quality, verified, downloads fast.
3. **WillHeld/hinglish_top** ✓: High-quality code-switched queries covering daily assistant interactions.
4. **festvox/cmu_hinglish_dog** ✓: Working conversational dialogue dataset configured as backup.

---

## 2. Funnel

*(Actual pipeline execution counts on the full 14,000 document raw corpus)*

| Stage | Input Count | Dropped Count | Output Count | Primary Reasons |
|-------|------------|---------------|--------------|-----------------|
| **Raw input** | 14,000 | 0 | 14,000 | 11,000 Devanagari Hindi + 3,000 Hinglish |
| **After normalize** | 14,000 | 1 | 13,999 | 1 document dropped for corrupted/garbled encoding |
| **After lang_detect** | 13,999 | 0 | 13,999 | Classified into: `hi` (10,970), `hi-Latn` (2,802), `en` (227), `other` (0) |
| **After quality** | 13,999 | 184 | 13,815 | Too short (<20 chars), boilerplate stubs, excessive repetitions |
| **After dedup (exact)** | 13,815 | 224 | 13,591 | Identical xxhash of normalized text |
| **After dedup (near MinHash LSH)** | 13,591 | 1,455 | 12,136 | MinHash LSH (128 perms, char 5-grams, Jaccard ≥ 0.80) |
| **After PII redaction** | 12,136 | 0 | 12,136 | 13 documents had sensitive phone/email/aadhaar redacted |
| **Eval set held out** | 12,136 | 200 | 11,936 | Deterministic SHA-256 hash sampling into `eval.jsonl` |
| **Contamination purge** | 11,936 | 1 | 11,935 | 1 training document near-duplicate with `eval_0061` purged |
| **Final Clean Training Corpus** | — | — | **11,935** | `hi`: 9,269 \| `hi-Latn`: 2,661 \| `en`: 205 |

---

## 3. Near-Duplicates

**Method**: MinHash LSH with 128 permutations, character 5-grams, Jaccard threshold ≥ 0.80.
**Performance optimization**: Pre-computing unique 5-grams and applying `MinHash.update_batch()` yielded a 20x throughput improvement over iterative token updates, allowing 14,000 full documents to be processed in ~97 seconds.

**Why 0.80 threshold**:
- **0.90+**: Near-identical boilerplate or articles differing by only single words/dates → duplicate.
- **0.80–0.90**: Same topic, identical structural templates, trivial variation → duplicate.
- **0.70–0.80**: Related topics with shared domain vocabulary (e.g., district stubs or awards in the same category) → substantively distinct, keep both.
- **Below 0.70**: Different content.

**Why character 5-grams**:
Hindi compound words and sandhi often blur word boundaries. Character n-grams are tokenizer-free, script-agnostic, and handle both Devanagari and Latin script seamlessly.

### Three pairs near the threshold (from real run)

1. **Pair at Jaccard 0.969 (DROPPED)**:
   - **Document A (`hindi_wiki.jsonl:10116`)**:
     > "फ़िल्मफ़ेयर सर्वश्रेष्ठ पार्श्व गायक पुरस्कार हिंदी फिल्मों के लिए वार्षिक फिल्मफेयर पुरस्कार के हिस्से के रूप में फिल्मफेयर द्वारा दिया जाने वाला पुरस्कार है। यह किसी पुरुष पार्श्व गायक को दिया जाता..."
   - **Document B (`hindi_wiki.jsonl:10117`)**:
     > "फ़िल्मफ़ेयर सर्वश्रेष्ठ पार्श्व गायिका पुरस्कार हिंदी फिल्मों के लिए वार्षिक फिल्मफेयर पुरस्कार के हिस्से के रूप में फिल्मफेयर द्वारा दिया जाने वाला पुरस्कार है। यह किसी महिला पार्श्व गायिका को दिया जाता..."
   - **Verdict**: Correctly dropped. The two articles are identical boilerplate templates with only "गायक/पुरुष" replaced by "गायिका/महिला".

2. **Pair at Jaccard 0.805 (DROPPED)**:
   - **Document A (`hindi_wiki.jsonl:5806`)**:
     > "अमेरिका का राष्ट्रिय पुस्तक पुरस्कार विजेता / अमेरिकी लेखक / 1905 में जन्मे लोग / १९७० में निधन"
   - **Document B (`hindi_wiki.jsonl:5809`)**:
     > "अमेरिका का राष्ट्रिय पुस्तक पुरस्कार विजेता / अमेरिकी लेखक / 1914 में जन्मे लोग / १९९७ में निधन"
   - **Verdict**: Correctly dropped. Category list stubs sharing 80.5% character overlap with only years changed.

3. **Pair at Jaccard 0.719 (KEPT)**:
   - **Document A (`hindi_wiki.jsonl:5806`)**:
     > "अमेरिका का राष्ट्रिय पुस्तक पुरस्कार विजेता / अमेरिकी लेखक / 1905 में जन्मे लोग / १९७० में निधन"
   - **Document B (`hindi_wiki.jsonl:5815`)**:
     > "अमेरिका का राष्ट्रिय पुस्तक पुरस्कार विजेता / अमेरिकी लेखक / 1932 में जन्मे लोग / २००९ में निधन"
   - **Verdict**: Correctly kept. Demonstrates that items with lower token-sharing (differing across multiple n-gram spans) fall below the 0.80 cutoff and remain in the corpus.

---

## 4. Filters

### Three documents that filters may have wrongly removed (Edge Cases)

1. **Short proverbs & idioms (Muhavare)**:
   - *Example*: "जो बोएगा सो काटेगा" (19 characters).
   - *Analysis*: Our threshold `MIN_CHARS = 20` would drop this single proverb if standalone. However, we included an `is_short_legitimate()` heuristic checking for terminal punctuation (`।`, `.`) and Devanagari script to rescue valid proverbs between 20-50 chars. Extremely short sub-20 character phrases were dropped to avoid isolated title fragments.

2. **Legitimate numerical & statistical tables**:
   - *Example*: Census tables or election result breakdowns containing district names and population numbers.
   - *Analysis*: We set the `mostly_digits` threshold to 70% (rather than 50%) specifically so that Hindi tabular data with Devanagari labels is preserved. Only pages that are >70% digits and punctuation are removed.

3. **Heavily code-mixed technical Hindi**:
   - *Example*: Programming tutorials or smartphone reviews written in Hindi but using English terms for 60%+ of tokens.
   - *Analysis*: Text with <30% Devanagari characters among alphabetic characters is routed to English (`en`) rather than `hi`. While the document is retained in the `en` partition of the training corpus, it is lost to the pure Hindi training slice.

---

## 5. Contamination

**Method**: Four-level hierarchical contamination check:
1. **Exact match**: Exact hash comparison of normalized text.
2. **Aggressive normalization**: Lowercase, strip all whitespace and punctuation, NFC normalize.
3. **Cross-script phonetic check**: Devanagari-to-Latin transliteration mapping compared against Latin-script training items.
4. **MinHash near-duplicate match**: MinHash Jaccard similarity ≥ 0.80 against eval items.

**Results from actual run**:
- **Eval items checked**: 200 documents (147 `hi`, 51 `hi-Latn`, 2 `en`).
- **Initial matches found**: 1 near-duplicate.
  - `eval_0061` matched training document `hi.jsonl.gz:1695` with Jaccard `0.805`.
  - Both were short district summary articles from Odisha ("जगतसिंहपुर ज़िला भारत के ओड़िशा राज्य का एक ज़िला है...").
- **Purge action**: The training document was automatically deleted from `output/clean/hi.jsonl.gz`.
- **Validation**: Re-running `src.contamination` on the purged corpus confirmed **0 contaminated eval items** across all 4 check levels.

**One candidate match rejected**:
- A training document discussing Indian history had Jaccard ~0.71 with an eval document on Indian culture. Because the shared n-grams consisted of standard background vocabulary (*"भारत"*, *"इतिहास"*, *"संस्कृति"*, *"प्राचीन"*), this was below the 0.80 threshold and correctly determined not to be contamination.

---

## 6. Reproducibility

To rebuild this exact dataset six months from now:

1. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Git Commit**:
   Tag or checkout commit: `3d808154f2ddae683be0f0c0924b336fea45b144`
3. **Download Raw Data**:
   ```bash
   ./download.sh raw/
   ```
   Or on Windows:
   ```powershell
   python download_data.py raw
   ```
   Input file hashes are recorded in `manifest.json`:
   - `hindi_wiki.jsonl`: `812f5a26e7e57f9b5b8edd370222c4785af54e126e50133e1bc13a3ae9cbfddd`
   - `hinglish.jsonl`: `7c87af9de1a1e24fb13a432d6d6ff30621ef7100bdee80e63f39ab89ec85034b`
4. **Run Pipeline**:
   ```bash
   ./run.sh raw/ output/
   ```
   Or on Windows:
   ```powershell
   .\run.ps1 raw output
   ```
   Every step uses deterministic hash-based sampling (`SHA-256`) and deterministic random seed `42`. Same input guaranteed produces bit-for-bit identical output.

---

## 7. Scale: 2 TB Architecture

**What breaks first at 2 TB:**
1. **In-Memory MinHash LSH Index**:
   At 14,000 documents, the LSH index uses ~50 MB RAM. At 2 TB (~500 million documents), in-memory LSH would require >2.5 TB of RAM.
   *Fix*:
   - Partition documents into language and domain buckets.
   - Use distributed MinHash with Redis/RocksDB or Apache Spark / Ray LSH.
   - Band-based hashing: partition by LSH bands (e.g. 16 bands of 8 rows) to find candidate duplicate pairs in a distributed MapReduce/Spark pass.
2. **Single-file Ledger**:
   A single `ledger.jsonl` would exceed 50 GB.
   *Fix*: Shard the ledger alongside the raw partition chunks (e.g. `ledger/part-*.parquet`).
3. **Streaming & Quality Filters**:
   The streaming generator, NFC normalization, PII regexes, and quality heuristics operate in $O(1)$ memory per document and scale horizontally across arbitrarily many CPU cores with zero shared state.

---

## 8. Wrap-up

### What I would do next (priority order)
1. **Perplexity-based LM filtering**: Train a small KenLM 5-gram language model on high-reputation Hindi text (NCERT / verified literature) to filter out low-perplexity gibberish and high-perplexity machine translation artifacts.
2. **Broader Hinglish crawling**: Scrape conversational Hinglish from public Reddit threads (`r/india`, `r/delhi`, `r/bollywood`) and YouTube comments to capture contemporary slang and code-mixing patterns.
3. **Parquet columnar storage**: Export clean corpus partitions as Snappy/ZSTD-compressed Parquet files with token count and quality metadata columns.
4. **Context-aware Named Entity Recognition (NER)**: Use an IndicNER model to replace regex-based PII redaction for personal names and physical addresses.
5. **Transliteration back-translation eval**: Include paired transliteration evaluation to assess the model's cross-script reasoning (Devanagari vs Latin).

### Time breakdown
- Architecture design & data discovery: 35 min
- Pipeline implementation & streaming ledger: 90 min
- Deduplication & MinHash batch optimization: 35 min
- Eval set generation & contamination checker: 30 min
- Verification, documentation & DECISIONS.md: 30 min
- **Total effort**: ~3.5 hours
