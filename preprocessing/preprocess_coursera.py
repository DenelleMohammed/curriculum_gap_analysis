#!/usr/bin/env python3
"""
preprocess_coursera.py — Preprocess scraped Coursera course data for SkillNER.

Input  : data/mooc/coursera_gap_courses.jsonl
Output : data/processed/coursera_courses_preprocessed.jsonl

Steps:
  1. Combine description + syllabus into one text
  2. Strip HTML and noise (Show less, Applied Learning Project, reviews)
  3. Normalize tech terms
  4. Remove non-alphanumeric characters (except safe symbols)
  5. Tokenize with spaCy (remove stopwords, punctuation, single chars)

Output record fields:
  topic_id, topic_label, search_query, course_title, provider, clean_text
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

from bs4 import BeautifulSoup
import spacy


# ==============================
# CONFIG
# ==============================

INPUT_FILE  = Path("data/mooc/coursera_gap_courses.jsonl")
OUTPUT_FILE = Path("data/processed/coursera_courses_preprocessed.jsonl")
SPACY_MODEL = "en_core_web_sm"


# ==============================
# TECH NORMALIZATIONS
# ==============================

_RAW_TECH_PATTERNS = [
    (r"\bc\s*\+\s*\+\b",                          "C++"),
    (r"\bc\s*#\b",                                 "C#"),
    (r"\bnode\s*\.?\s*js\b",                       "Node.js"),
    (r"\breact\s*\.?\s*js\b",                      "React.js"),
    (r"\bci\s*\/\s*cd\b",                          "CI/CD"),
    (r"\bmachine[-\s]?learning\b",                 "machine learning"),
    (r"\bdeep[-\s]?learning\b",                    "deep learning"),
    (r"\bnatural[-\s]?language[-\s]?processing\b", "natural language processing"),
    (r"\bcomputer[-\s]?vision\b",                  "computer vision"),
    (r"\bartificial\s+intelligence\b",             "artificial intelligence"),
    (r"\btcp\s*\/\s*ip\b",                         "TCP/IP"),
    (r"\bmy\s*sql\b",                              "MySQL"),
    (r"\bmongo\s*db\b",                            "MongoDB"),
    (r"\bpower\s*bi\b",                            "Power BI"),
    (r"\bno\s*sql\b",                              "NoSQL"),
    (r"\baws\b",                                   "AWS"),
    (r"\bgcp\b",                                   "GCP"),
    (r"\bmlops\b",                                 "MLOps"),
    (r"\betl\b",                                   "ETL"),
    (r"\bapi\b",                                   "API"),
    (r"\brag\b",                                   "RAG"),
    (r"\bllm\b",                                   "LLM"),
]

TECH_NORMALIZATIONS = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _RAW_TECH_PATTERNS
]

# Noise patterns to strip before cleaning
_NOISE_PATTERNS = [
    re.compile(r"show less.*$",                    re.IGNORECASE | re.DOTALL),
    re.compile(r"show more.*$",                    re.IGNORECASE | re.DOTALL),
    re.compile(r"applied learning project.*$",     re.IGNORECASE | re.DOTALL),
    re.compile(r"acknowledgements?:.*$",           re.IGNORECASE | re.DOTALL),
    re.compile(r"acknowledgment:.*$",              re.IGNORECASE | re.DOTALL),
    # Remove reviewer quotes (short sentences after "Show less" stripped above)
    re.compile(r"\b(excellent|great|good|outstanding|i (really|liked|found|gained))[^.]*\.", re.IGNORECASE),
]

_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_MULTISPACE_RE  = re.compile(r"\s+")
_ALLOWED_RE     = re.compile(r"[^a-zA-Z0-9\s\+\#\.\-\/_]")


# ==============================
# IO
# ==============================

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ==============================
# TEXT BUILDING
# ==============================

def build_raw_text(rec: Dict[str, Any]) -> str:
    """Combine description and syllabus into one raw text."""
    parts = []

    desc = (rec.get("description") or "").strip()
    if desc:
        parts.append(desc)

    syllabus = rec.get("syllabus") or []
    if isinstance(syllabus, list) and syllabus:
        # Join syllabus module titles as a sentence-like string
        parts.append(". ".join(syllabus))

    return " ".join(parts)


# ==============================
# CLEANING
# ==============================

def strip_noise(text: str) -> str:
    """Remove Coursera boilerplate, reviews, and structural noise."""
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    return text


def clean_text(raw: str) -> str:
    if not raw:
        return ""

    # Strip HTML
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    text = _HTML_TAG_RE.sub(" ", text)

    # Remove noise
    text = strip_noise(text)

    # Normalize tech terms
    for pattern, repl in TECH_NORMALIZATIONS:
        text = pattern.sub(repl, text)

    # Remove unsafe characters
    text = _ALLOWED_RE.sub(" ", text)

    # Collapse whitespace
    text = _MULTISPACE_RE.sub(" ", text).strip()

    return text


# ==============================
# TOKENIZATION
# ==============================

try:
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
except OSError:
    raise SystemExit(f"Run: python -m spacy download {SPACY_MODEL}")


def tokenize(text: str) -> List[str]:
    """Remove stopwords, punctuation, and single-character tokens."""
    doc = nlp(text)
    return [
        t.text for t in doc
        if not t.is_stop and not t.is_punct and len(t.text) >= 2
    ]


# ==============================
# MAIN
# ==============================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE.resolve()}")

    records = []

    print("[INFO] Preprocessing MOOC courses...")

    for rec in iter_jsonl(INPUT_FILE):
        topic_id     = rec.get("topic_id")
        topic_label  = rec.get("topic_label", "")
        search_query = rec.get("search_query", "")
        course_title = rec.get("course_title", "")
        provider     = rec.get("provider", "")
        # url, platform, description, syllabus are used only to build clean_text

        raw_text   = build_raw_text(rec)
        cleaned    = clean_text(raw_text)
        tokens     = tokenize(cleaned)
        clean_text_str = " ".join(tokens)

        records.append({
            "topic_id":    topic_id,
            "topic_label": topic_label,
            "search_query": search_query,
            "course_title": course_title,
            "provider":    provider,
            "clean_text":  clean_text_str,
        })

        print(f"  Topic {str(topic_id):<3} | {course_title[:50]:<50} | {len(tokens)} tokens")

    write_jsonl(OUTPUT_FILE, records)

    print()
    print("=" * 60)
    print(f"  Courses preprocessed : {len(records)}")
    print(f"  Output               : {OUTPUT_FILE.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()