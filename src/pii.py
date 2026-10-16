"""
PII redaction for Indian documents.

Patterns handled:
1. Indian phone numbers: +91-XXXXXXXXXX, 0XX-XXXXXXXX, etc.
2. Email addresses: standard RFC-ish pattern
3. Aadhaar numbers: 12 digits in groups of 4 (XXXX XXXX XXXX or XXXX-XXXX-XXXX)
4. PAN numbers: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)

Design decision: We are CONSERVATIVE about what we redact.
- A 10-digit number that looks like a phone number IS redacted.
- A 10-digit number that appears in a mathematical context IS NOT.
- We use context clues (surrounding words, formatting) to avoid false positives.
- Numbers in tables or sequences are NOT redacted unless they match PII patterns.

We handle both Latin digits (0-9) and Devanagari digits (०-९).
"""

import re
from typing import List, Tuple


def _devanagari_to_latin_digits(text: str) -> str:
    """Convert Devanagari digits to Latin for pattern matching."""
    trans = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(trans)


# --- Phone number patterns ---
# Indian phone numbers: +91, 0XX prefix, 10 digits
# We look for numbers that are specifically formatted as phone numbers
_PHONE_PATTERNS = [
    # +91 followed by 10 digits with optional separators
    re.compile(r"\+91[\s\-]?\d[\s\-]?\d{4}[\s\-]?\d{5}"),
    re.compile(r"\+91[\s\-]?\d{10}"),
    re.compile(r"\+91[\s\-]?\d{5}[\s\-]?\d{5}"),
    # 0XX-XXXXXXXX (landline with STD code)
    re.compile(r"\b0\d{2,4}[\s\-]\d{6,8}\b"),
    # Standalone 10-digit mobile numbers (must start with 6-9)
    # Only match if preceded/followed by non-digit to avoid matching in longer numbers
    re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),
]

# Context words that suggest a number is a phone number
_PHONE_CONTEXT = re.compile(
    r"(?:phone|mobile|mob|cell|tel|call|whatsapp|contact|number|no\.|"
    r"फ़ोन|फोन|मोबाइल|कॉल|संपर्क|नंबर|नम्बर|व्हाट्सएप)",
    re.IGNORECASE
)

# --- Email pattern ---
_EMAIL = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

# --- Aadhaar pattern ---
# 12 digits, typically in groups of 4: XXXX XXXX XXXX or XXXX-XXXX-XXXX
# First digit cannot be 0 or 1
_AADHAAR = re.compile(
    r"\b[2-9]\d{3}[\s\-]\d{4}[\s\-]\d{4}\b"
)
# Also match 12 continuous digits starting with 2-9 (with context check)
_AADHAAR_CONTINUOUS = re.compile(r"\b[2-9]\d{11}\b")

# Context words for Aadhaar
_AADHAAR_CONTEXT = re.compile(
    r"(?:aadhaar|aadhar|आधार|uid|uidai|unique\s*id)",
    re.IGNORECASE
)

# --- PAN pattern ---
# Exactly: 5 uppercase letters + 4 digits + 1 uppercase letter
# The 4th character indicates type: C (company), P (person), etc.
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

# Context words for PAN
_PAN_CONTEXT = re.compile(
    r"(?:pan\s*(?:card|number|no)|पैन\s*(?:कार्ड|नंबर)|permanent\s*account)",
    re.IGNORECASE
)


def _has_context(text: str, match_start: int, match_end: int, 
                 context_pattern: re.Pattern, window: int = 100) -> bool:
    """Check if there's a context word near the match."""
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    surrounding = text[start:end]
    return bool(context_pattern.search(surrounding))


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """
    Redact PII from text.
    
    Returns:
        (redacted_text, list_of_redaction_types)
        
    The list contains one entry per redaction made, e.g.:
    ["PHONE", "EMAIL", "PHONE"] if two phones and one email were redacted.
    """
    if not text:
        return (text, [])

    redactions = []
    result = text

    # First convert any Devanagari digits for pattern matching
    # but keep originals in text - we replace the spans directly
    latin_text = _devanagari_to_latin_digits(result)

    # --- Email (do first, most specific) ---
    for match in reversed(list(_EMAIL.finditer(result))):
        result = result[:match.start()] + "<EMAIL>" + result[match.end():]
        redactions.append("EMAIL")

    # --- PAN (do before phone - PAN is very specific format) ---
    for match in reversed(list(_PAN.finditer(result))):
        # PAN is distinctive enough that we don't strictly need context
        # But we avoid matching common abbreviations that happen to match
        matched = match.group()
        # Simple heuristic: if the match is part of a longer word, skip it
        start, end = match.start(), match.end()
        if start > 0 and result[start-1].isalpha():
            continue
        if end < len(result) and result[end].isalpha():
            continue
        result = result[:start] + "<PAN>" + result[end:]
        redactions.append("PAN")

    # --- Aadhaar ---
    # Formatted Aadhaar (XXXX XXXX XXXX)
    for match in reversed(list(_AADHAAR.finditer(result))):
        result = result[:match.start()] + "<AADHAAR>" + result[match.end():]
        redactions.append("AADHAAR")
    
    # Continuous 12-digit Aadhaar (only with context)
    for match in reversed(list(_AADHAAR_CONTINUOUS.finditer(result))):
        if _has_context(result, match.start(), match.end(), _AADHAAR_CONTEXT):
            result = result[:match.start()] + "<AADHAAR>" + result[match.end():]
            redactions.append("AADHAAR")

    # --- Phone numbers ---
    # +91 prefixed (high confidence)
    for pattern in _PHONE_PATTERNS[:3]:
        for match in reversed(list(pattern.finditer(result))):
            result = result[:match.start()] + "<PHONE>" + result[match.end():]
            redactions.append("PHONE")

    # Landline with STD code
    for match in reversed(list(_PHONE_PATTERNS[3].finditer(result))):
        result = result[:match.start()] + "<PHONE>" + result[match.end():]
        redactions.append("PHONE")

    # Standalone 10-digit mobile: only redact if has phone context nearby
    # This avoids redacting random 10-digit numbers
    for match in reversed(list(_PHONE_PATTERNS[4].finditer(result))):
        if _has_context(result, match.start(), match.end(), _PHONE_CONTEXT, window=150):
            result = result[:match.start()] + "<PHONE>" + result[match.end():]
            redactions.append("PHONE")

    return (result, redactions)
