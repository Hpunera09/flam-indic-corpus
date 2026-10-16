"""
Language detection based on Unicode script analysis.

We do NOT rely on source labels. Instead we classify by examining the actual
characters in the text.

Classification rules:
- hi:      Majority Devanagari script. Allows mixed English words (common in
           modern Hindi writing). Threshold: ≥30% Devanagari chars among
           alphabetic characters.
- hi-Latn: Hindi/Hinglish written in Latin script. Detected by presence of
           common Hindi words transliterated into Latin. We use a vocabulary
           of high-frequency Hindi words in Latin script.
- en:      Predominantly English/Latin with no significant Hindi markers.
- other:   Anything else (other Indic scripts, CJK, etc.)

The 30% Devanagari threshold is deliberately low to catch Hindi text that
has substantial English mixing (e.g., technical content written mostly in
Hindi but with English terms).
"""

import re
import unicodedata
from typing import Tuple

# Devanagari Unicode range: U+0900 to U+097F
# Extended Devanagari: U+A8E0 to U+A8FF, U+1CD0 to U+1CFF (Vedic extensions)
_DEVANAGARI_RANGE = set(range(0x0900, 0x0980)) | set(range(0xA8E0, 0xA900))

# Common Hindi words in Latin script (transliterated)
# These are high-frequency function words and common terms that strongly
# indicate Hindi/Hinglish content when written in Latin script.
_HINGLISH_MARKERS = {
    # Function words
    "hai", "hain", "ho", "tha", "thi", "the", "hoga", "hogi",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "pe",
    "ye", "yeh", "wo", "woh", "wala", "wali", "wale",
    "kya", "kab", "kaise", "kahan", "kyun", "kyunki",
    "nahi", "nhi", "nahin", "mat", "na",
    "aur", "ya", "lekin", "magar", "par", "phir",
    "ek", "do", "teen", "char",
    "bhi", "hi", "to", "toh", "bhai",
    "agar", "jab", "tab", "abhi", "ab",
    "hum", "tum", "main", "mera", "meri", "mere",
    "tera", "teri", "tere", "uska", "uski", "uske",
    "apna", "apni", "apne",
    "kuch", "bahut", "bohot", "zyada", "kam",
    "accha", "acha", "achha", "bura", "sahi", "galat",
    "kaam", "ghar", "log", "baat", "din", "raat",
    "kar", "karo", "karna", "raha", "rahi", "rahe",
    "jaa", "jao", "jana", "aana", "aao",
    "dekho", "dekh", "suno", "sun", "bolo", "bol",
    "sab", "sabko", "sabse", "dono",
    "yaar", "arre", "abe", "oye",
    "achanak", "bilkul", "zaroor", "shayad",
    "liye", "liya", "diya", "kiya",
    "wapis", "sath", "saath", "baad",
    # Common verbs / phrases
    "samajh", "samjho", "pata", "chalo", "chal",
    "theek", "thik", "shukriya", "dhanyawad",
    "paisa", "rupay", "rupee",
}

# Minimum text length for reliable detection
_MIN_LENGTH_FOR_DETECTION = 10


def _count_scripts(text: str) -> dict:
    """Count characters by Unicode script category."""
    counts = {
        "devanagari": 0,
        "latin": 0,
        "other_alpha": 0,
        "digit": 0,
        "punct_space": 0,
    }
    for ch in text:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cp in _DEVANAGARI_RANGE:
            counts["devanagari"] += 1
        elif cat.startswith("L"):
            # It's a letter but not Devanagari
            if "LATIN" in unicodedata.name(ch, ""):
                counts["latin"] += 1
            else:
                counts["other_alpha"] += 1
        elif cat.startswith("N"):
            counts["digit"] += 1
        else:
            counts["punct_space"] += 1
    return counts


def _count_hinglish_markers(text: str) -> int:
    """Count how many Hinglish marker words appear in text."""
    # Tokenize roughly on whitespace and punctuation
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return sum(1 for w in words if w in _HINGLISH_MARKERS)


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect language of text.
    
    Returns:
        (lang, confidence) where lang is one of 'hi', 'hi-Latn', 'en', 'other'
        and confidence is a float 0-1.
    """
    if not text or len(text.strip()) < _MIN_LENGTH_FOR_DETECTION:
        return ("other", 0.0)

    counts = _count_scripts(text)
    total_alpha = counts["devanagari"] + counts["latin"] + counts["other_alpha"]

    if total_alpha == 0:
        return ("other", 0.0)

    dev_ratio = counts["devanagari"] / total_alpha
    latin_ratio = counts["latin"] / total_alpha
    other_ratio = counts["other_alpha"] / total_alpha

    # Rule 1: If substantial Devanagari, it's Hindi
    # 30% threshold allows for Hindi text with heavy English code-mixing
    if dev_ratio >= 0.30:
        confidence = min(1.0, dev_ratio + 0.2)
        return ("hi", confidence)

    # Rule 2: If mostly Latin script, check for Hinglish markers
    if latin_ratio >= 0.50:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        total_words = len(words)
        if total_words > 0:
            hinglish_count = _count_hinglish_markers(text)
            hinglish_ratio = hinglish_count / total_words

            # If ≥15% of words are Hindi markers, classify as Hinglish
            # This threshold catches code-mixed text while avoiding
            # false positives on English text that happens to contain
            # words like "hi" or "to" (which are also English words)
            if hinglish_ratio >= 0.15:
                return ("hi-Latn", min(1.0, hinglish_ratio * 3))

            # Lower threshold if there are multiple distinct markers
            distinct_markers = len(set(w for w in words if w in _HINGLISH_MARKERS))
            if distinct_markers >= 5 and hinglish_ratio >= 0.08:
                return ("hi-Latn", min(1.0, hinglish_ratio * 3))

        return ("en", min(1.0, latin_ratio))

    # Rule 3: Other scripts dominate
    if other_ratio > 0.5:
        return ("other", other_ratio)

    # Fallback: small amounts of Devanagari with some Latin
    # Could be Hindi with lots of English or vice versa
    if dev_ratio >= 0.10:
        return ("hi", dev_ratio + 0.1)

    return ("other", 0.5)
