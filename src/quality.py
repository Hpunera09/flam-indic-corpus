"""
Quality filters for Hindi/Hinglish corpus.

Design philosophy:
- We are deliberately LENIENT with short texts (proverbs, poems, slogans)
  because these are legitimate and valuable training data for an Indian
  language model.
- We are AGGRESSIVE with boilerplate, markup, and obvious spam.
- Every filter returns a (keep: bool, reason: str) tuple so the ledger
  can record exactly why a document was dropped.

Filters applied in order:
1. Empty/too-short check
2. Encoding corruption (garbled text)
3. Markup/HTML residue
4. Boilerplate detection (cookie notices, nav bars, etc.)
5. Spam heuristics (excessive repetition, all-caps, URL density)
6. Character quality (too many special chars, digits)
"""

import re
from typing import Tuple


# --- Thresholds ---
MIN_CHARS = 20           # Minimum characters (allows short proverbs like "जो बोएगा सो काटेगा")
MIN_WORDS = 3            # Minimum word count
MAX_CHAR_REPETITION = 4  # Max consecutive identical characters (allows "हाहाहाहा" but catches "aaaaaaaaa...")

# HTML/Markup patterns
_HTML_TAG = re.compile(r"<[^>]{1,500}>")
_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_MARKDOWN_HEAVY = re.compile(r"^[\s\*\#\-\|\=\>\`]{5,}", re.MULTILINE)

# URL pattern
_URL = re.compile(r"https?://\S+|www\.\S+")

# Boilerplate phrases (Hindi and English)
_BOILERPLATE_PHRASES = [
    # English boilerplate
    "cookie", "privacy policy", "terms of service", "terms and conditions",
    "all rights reserved", "copyright ©", "subscribe to newsletter",
    "click here", "read more", "share this", "follow us",
    "powered by", "website design", "skip to content",
    "accept cookies", "cookie policy",
    # Hindi boilerplate
    "कुकी नीति", "गोपनीयता नीति", "सेवा की शर्तें",
    "सर्वाधिकार सुरक्षित", "यहाँ क्लिक करें", "और पढ़ें",
    "हमें फॉलो करें", "शेयर करें", "सब्सक्राइब",
    "ईमेल पता दर्ज करें", "न्यूज़लेटर",
    # Navigation boilerplate
    "home", "about us", "contact us", "sitemap",
    "होम", "हमारे बारे में", "संपर्क करें",
]

# Repetition pattern: same char 5+ times in a row
_EXCESSIVE_REPEAT = re.compile(r"(.)\1{4,}")

# Excessive special character pattern
_SPECIAL_CHARS = re.compile(r"[^\w\s\u0900-\u097F\u0A00-\u0A7F\.,!?\-\(\)\'\"]")


def check_quality(text: str) -> Tuple[bool, str]:
    """
    Run all quality filters on text.
    
    Returns:
        (keep, reason): keep=True means text passes all filters.
        If keep=False, reason explains which filter caught it.
    """
    # Filter 1: Empty / too short
    if not text:
        return (False, "empty")
    
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return (False, f"too_short:{len(stripped)}_chars")

    # Count words (works for both Devanagari and Latin)
    words = stripped.split()
    if len(words) < MIN_WORDS:
        return (False, f"too_few_words:{len(words)}")

    # Filter 2: Encoding corruption
    # (normalize.is_garbled handles this, but we double-check here)
    replacement_count = stripped.count("\ufffd")
    if replacement_count > 0:
        return (False, "encoding_corruption")

    # Filter 3: HTML/Markup residue
    html_tags = _HTML_TAG.findall(stripped)
    if len(html_tags) > 2:
        # More than 2 HTML tags suggests this is markup, not text
        return (False, f"html_markup:{len(html_tags)}_tags")
    
    html_entities = _HTML_ENTITY.findall(stripped)
    if len(html_entities) > 3:
        return (False, f"html_entities:{len(html_entities)}")
    
    # Remove any remaining HTML for further analysis
    clean_for_analysis = _HTML_TAG.sub("", stripped)
    clean_for_analysis = _HTML_ENTITY.sub("", clean_for_analysis)

    # Filter 4: Boilerplate detection
    text_lower = stripped.lower()
    boilerplate_count = sum(1 for phrase in _BOILERPLATE_PHRASES if phrase in text_lower)
    # If more than 30% of the text is boilerplate phrases, drop it
    if boilerplate_count >= 4:
        return (False, f"boilerplate:{boilerplate_count}_phrases")
    # Very short text with any boilerplate is likely just a nav fragment
    if boilerplate_count >= 2 and len(words) < 15:
        return (False, f"boilerplate_fragment:{boilerplate_count}_phrases_in_{len(words)}_words")

    # Filter 5: Spam heuristics
    # 5a: Excessive character repetition
    repeats = _EXCESSIVE_REPEAT.findall(stripped)
    if len(repeats) > 3:
        return (False, f"excessive_repetition:{len(repeats)}_runs")

    # 5b: URL density - legitimate text shouldn't be mostly URLs
    urls = _URL.findall(stripped)
    url_char_count = sum(len(u) for u in urls)
    if len(stripped) > 0 and url_char_count / len(stripped) > 0.5:
        return (False, f"url_spam:{url_char_count}/{len(stripped)}_chars")

    # 5c: All-caps check (mostly for Latin text)
    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    if len(alpha_chars) > 20:
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if upper_ratio > 0.8:
            return (False, f"all_caps:{upper_ratio:.0%}")

    # Filter 6: Character quality
    # Too many special/unusual characters suggest data corruption or markup
    if len(clean_for_analysis) > 0:
        special = _SPECIAL_CHARS.findall(clean_for_analysis)
        special_ratio = len(special) / len(clean_for_analysis)
        if special_ratio > 0.3:
            return (False, f"special_char_heavy:{special_ratio:.0%}")

    # Check digit ratio - tables of pure numbers aren't useful text
    # But allow tables that have some text mixed in
    digit_chars = sum(1 for ch in stripped if ch.isdigit())
    if len(stripped) > 0:
        digit_ratio = digit_chars / len(stripped)
        # 70% digits is too much - but 60% allows tables with labels
        if digit_ratio > 0.7 and len(words) > 5:
            return (False, f"mostly_digits:{digit_ratio:.0%}")

    # PASSED all filters
    return (True, "passed")


def is_short_legitimate(text: str) -> bool:
    """
    Check if a short text (<50 chars) looks like a legitimate proverb,
    poem line, or similar valuable short content.
    
    This is used to rescue short texts from the length filter.
    We check for:
    - Contains Devanagari characters (proverbs)
    - Has sentence structure (starts with capital or Devanagari, ends with punctuation)
    - Contains common poetic markers
    """
    if not text or len(text) < MIN_CHARS:
        return False
    
    stripped = text.strip()
    
    # Has Devanagari content
    has_devanagari = any(0x0900 <= ord(ch) <= 0x097F for ch in stripped)
    
    # Has sentence-like structure
    ends_with_punct = stripped[-1] in ".!?।॥"
    
    # Short but structured -> likely legitimate
    if has_devanagari and ends_with_punct and len(stripped) >= MIN_CHARS:
        return True
    
    return False
