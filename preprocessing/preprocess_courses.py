#!/usr/bin/env python3
"""
Course description preprocessor for SkillNER + LDA pipeline.

Source handled:
  - Course catalog : outcomes_and_content.json   (JSON array)

Output:
  - data/processed/all_courses_preprocessed.jsonl

Each output record contains:
  course_code | course code as listed (e.g. "COMP 1600")
  skills      | SkillNER-friendly cleaned + tokenized text, built by combining
                description, rationale, aims, learning_outcomes and course_content

Cleaning steps (applied to all sources, same as preprocess_jobs.py):
  1. Strip HTML tags
  2. Fix letter-spaced words (e.g. "c o d e" -> "code")
  3. Normalize tech variants to canonical SkillNER surface forms
  4. Remove characters outside [a-zA-Z0-9 + # . / - _]
  5. Normalize whitespace
  6. Tokenize with spaCy (remove stopwords, punctuation, single chars)

Deduplication:
  - Content hash (MD5) deduplication, matching preprocess_jobs.py
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup
import spacy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_FILE  = Path("outcomes_and_content.json")
OUTPUT_DIR  = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "all_courses_preprocessed.jsonl"


# ---------------------------------------------------------------------------
# Text normalisation (identical rules to preprocess_jobs.py)
# ---------------------------------------------------------------------------

_SPELLED_OUT_RE   = re.compile(r"(?:\b[a-z]\s+){3,}[a-z]\b", flags=re.IGNORECASE)
_HTML_TAG_RE      = re.compile(r"<[^>]+>")
_MULTISPACE_RE    = re.compile(r"\s+")
_ALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s\+\#\.\-\/_]")

_RAW_TECH_PATTERNS = [
    # C-family
    (r"\bc\s*\+\s*\+\b",          "C++"),
    (r"\bc\s*#\b",                 "C#"),
    (r"\bf\s*#\b",                 "F#"),
    (r"\bobjective\s*-\s*c\b",    "Objective-C"),
    (r"\bobjective\s+c\b",        "Objective-C"),
    # .NET
    (r"\bdot\s*net\b",            ".NET"),
    (r"\basp\s*\.?\s*net\b",      "ASP.NET"),
    # JS ecosystem
    (r"\bnode\s*\.?\s*js\b",      "Node.js"),
    (r"\breact\s*\.?\s*js\b",     "React.js"),
    (r"\bvue\s*\.?\s*js\b",       "Vue.js"),
    (r"\bnext\s*\.?\s*js\b",      "Next.js"),
    (r"\bnuxt\s*\.?\s*js\b",      "Nuxt.js"),
    # DevOps
    (r"\bci\s*\/\s*cd\b",         "CI/CD"),
    (r"\bdev\s*\/\s*sec\s*\/\s*ops\b", "DevSecOps"),
    # AI/ML phrases
    (r"\bmachine[-\s]?learning\b",              "machine learning"),
    (r"\bdeep[-\s]?learning\b",                 "deep learning"),
    (r"\breinforcement[-\s]?learning\b",        "reinforcement learning"),
    (r"\bnatural[-\s]?language[-\s]?processing\b", "natural language processing"),
    (r"\bcomputer[-\s]?vision\b",               "computer vision"),
    (r"\bartificial intelligence\b",            "artificial intelligence"),
    # Cloud
    (r"\bamazon web services\b",   "AWS"),
    (r"\bgoogle cloud platform\b", "GCP"),
    (r"\bmicrosoft azure\b",       "Azure"),
    # Databases
    (r"\bpostgre\s*sql\b",         "PostgreSQL"),
    (r"\bpostgres\s*sql\b",        "PostgreSQL"),
    (r"\bmy\s*sql\b",              "MySQL"),
    (r"\bmongo\s*db\b",            "MongoDB"),
    (r"\bno\s*sql\b",              "NoSQL"),
    # Networking
    (r"\btcp\s*\/\s*ip\b",        "TCP/IP"),
    (r"\bwi[\-\s]?fi\b",          "Wi-Fi"),
]

TECH_NORMALIZATIONS = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _RAW_TECH_PATTERNS
]


def _fix_spelled_out(text: str) -> str:
    """Convert 'c o d e' -> 'code' style noise."""
    return _SPELLED_OUT_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def _normalize_tech(text: str) -> str:
    for pattern, repl in TECH_NORMALIZATIONS:
        text = pattern.sub(repl, text)
    return text


def clean_text(raw: str) -> str:
    """
    Returns SkillNER-friendly cleaned text (not yet tokenized).
    Preserves: C++, C#, .NET, Node.js, CI/CD, TCP/IP, Wi-Fi, etc.
    """
    if not raw:
        return ""
    # 1. Strip HTML (BeautifulSoup handles malformed markup better than regex alone)
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    # Fallback regex pass for any residual tags
    text = _HTML_TAG_RE.sub(" ", text)
    # 2. Fix letter-spaced words
    text = _fix_spelled_out(text)
    # 3. Canonical tech forms
    text = _normalize_tech(text)
    # 4. Drop characters outside the safe set
    text = _ALLOWED_CHARS_RE.sub(" ", text)
    # 5. Collapse whitespace
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tokenisation (spaCy)
# ---------------------------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    print("[WARN] 'en_core_web_sm' not available; falling back to blank 'en' model.")
    nlp = spacy.blank("en")


def tokenize(text: str) -> List[str]:
    """Remove stopwords, punctuation, and single-character tokens."""
    doc = nlp(text)
    return [
        t.text for t in doc
        if not t.is_stop and not t.is_punct and len(t.text) >= 2
    ]


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_courses(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"[WARN] Expected a JSON array in {path.name}, got {type(data).__name__}")
                return []
            return data
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {path.name}: {e}")
            return []


def build_raw_skills_text(rec: Dict[str, Any]) -> str:
    """
    Combines description, rationale, aims, learning_outcomes and course_content
    into a single raw text blob prior to cleaning/tokenizing.
    """
    parts: List[str] = []

    for field in ("description", "rationale", "aims"):
        value = rec.get(field)
        if value:
            parts.append(str(value))

    learning_outcomes = rec.get("learning_outcomes") or []
    parts.extend(str(item) for item in learning_outcomes if item)

    course_content = rec.get("course_content") or {}
    for topic, items in course_content.items():
        if topic:
            parts.append(str(topic))
        for item in items or []:
            if item:
                parts.append(str(item))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE.resolve()}")
        return

    print(f"[INFO] Loading course catalog ({INPUT_FILE.name})...")
    courses = load_courses(INPUT_FILE)

    seen_hashes: set = set()

    total_raw = len(courses)
    total_out = 0
    skipped_missing  = 0
    skipped_hash_dup = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:

        for rec in courses:
            course_code = rec.get("course_code")

            # --- Must have some combinable text ---
            raw_text = build_raw_skills_text(rec).strip()
            if not raw_text:
                skipped_missing += 1
                continue

            # --- Clean + tokenize ---
            cleaned = clean_text(raw_text)
            tokens  = tokenize(cleaned)
            if not tokens:
                skipped_missing += 1
                continue

            skills_str = " ".join(tokens)

            # --- Content hash dedup ---
            content_hash = hashlib.md5(skills_str.encode()).hexdigest()
            if content_hash in seen_hashes:
                skipped_hash_dup += 1
                continue
            seen_hashes.add(content_hash)

            record = {
                "course_code": course_code or None,
                "skills":      skills_str,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_out += 1

            if total_out % 200 == 0:
                print(f"[INFO] Written {total_out} records so far...")

    print()
    print("=" * 50)
    print(f"  Raw records seen      : {total_raw}")
    print(f"  Written to output     : {total_out}")
    print(f"  Skipped (content dup) : {skipped_hash_dup}")
    print(f"  Skipped (empty/bad)   : {skipped_missing}")
    print(f"  Output file           : {OUTPUT_FILE.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
