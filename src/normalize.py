"""
Unicode normalization for Devanagari and Latin text.

Design decisions:
- NFC normalization: Devanagari has composed and decomposed forms. NFC ensures
  that e.g. क + ् + ष  == क्ष when they are the same character.
- Remove invisible/zero-width characters that are not semantically meaningful
  (ZWJ/ZWNJ are kept because they affect Devanagari rendering).
- Normalize whitespace: collapse runs, strip control chars except newlines.
- Handle common encoding artifacts: replacement chars, BOM, etc.
"""

import re
import unicodedata


# Zero-width chars to REMOVE (not ZWJ/ZWNJ which are meaningful in Devanagari)
_INVISIBLE_CHARS = re.compile(
    "["
    "\u00ad"          # soft hyphen
    "\u200b"          # zero-width space
    "\u200e\u200f"    # LTR/RTL marks
    "\u202a-\u202e"   # bidi embedding/override
    "\u2060"          # word joiner
    "\u2061-\u2064"   # invisible operators
    "\ufeff"          # BOM / zero-width no-break space
    "\ufff9-\ufffb"   # interlinear annotations
    "\U000e0001"      # language tag
    "\U000e0020-\U000e007f"  # tag characters
    "]"
)

# Whitespace normalization: collapse all Unicode whitespace (except \n) to single space
_WHITESPACE = re.compile(r"[^\S\n]+")  # matches whitespace except newline
_BLANK_LINES = re.compile(r"\n{3,}")   # collapse 3+ newlines to 2

# Common encoding artifacts
_REPLACEMENT_CHAR = "\ufffd"


def normalize_text(text: str) -> str:
    """
    Normalize a text string for corpus use.
    
    Steps:
    1. NFC Unicode normalization (critical for Devanagari)
    2. Remove invisible characters (keep ZWJ/ZWNJ)
    3. Normalize whitespace
    4. Strip leading/trailing whitespace per line
    5. Remove control characters (keep \n, \t)
    
    Returns normalized text, or empty string if input is essentially empty.
    """
    if not text:
        return ""

    # Step 1: NFC normalization - this is the most important step for Devanagari
    # It ensures composed forms: e.g. Ka + Nukta = Qa
    text = unicodedata.normalize("NFC", text)

    # Step 2: Remove invisible formatting characters
    # We keep ZWJ (\u200d) and ZWNJ (\u200c) because they are semantically
    # meaningful in Devanagari (e.g., conjuncts vs explicit halant display)
    text = _INVISIBLE_CHARS.sub("", text)

    # Step 3: Remove control characters except \n, \t, \r
    # This catches things like null bytes, bell, form feed, etc.
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t", "\r") or not unicodedata.category(ch).startswith("C")
        # Keep Cf (format) chars that weren't removed above - this preserves ZWJ/ZWNJ
        or unicodedata.category(ch) == "Cf"
    )

    # Step 4: Normalize whitespace
    # - Convert \r\n and \r to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # - Collapse horizontal whitespace (tabs, spaces, etc.) to single space
    text = _WHITESPACE.sub(" ", text)
    # - Collapse excessive blank lines (3+ newlines -> 2)
    text = _BLANK_LINES.sub("\n\n", text)

    # Step 5: Strip each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Step 6: Strip overall
    text = text.strip()

    return text


def has_encoding_issues(text: str) -> bool:
    """Check if text has signs of encoding corruption."""
    if not text:
        return False
    # Replacement character is a clear sign of encoding issues
    if _REPLACEMENT_CHAR in text:
        return True
    # Mojibake patterns: Latin chars that look like decoded-wrong Devanagari
    # Common pattern: seeing Ã, Â sequences (UTF-8 decoded as Latin-1)
    mojibake_count = sum(1 for ch in text if ch in "ÃÂ")
    if mojibake_count > len(text) * 0.02:  # more than 2% mojibake chars
        return True
    return False


def is_garbled(text: str) -> bool:
    """
    Check if text appears to be garbled/corrupted encoding.
    More aggressive than has_encoding_issues - also checks for
    excessive use of characters that shouldn't appear in natural text.
    """
    if has_encoding_issues(text):
        return True
    if not text:
        return True
    # Check for excessive non-printable characters
    non_printable = sum(
        1 for ch in text
        if unicodedata.category(ch).startswith("C") and ch not in "\n\t"
    )
    if non_printable > len(text) * 0.05:
        return True
    return False
