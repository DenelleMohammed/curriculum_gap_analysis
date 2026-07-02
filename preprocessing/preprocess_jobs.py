#!/usr/bin/env python3
"""
Unified job description preprocessor for SkillNER + LDA pipeline.

Sources handled:
  - LinkedIn       : data/raw/linkedin_jobs.jsonl          (JSONL format)
  - Workopolis     : data/raw/workopolis_jobs.json         (JSON array)
  - Reed UK        : data/raw/reed_jobs.json               (JSON array)

Output:
  - data/processed/all_jobs_preprocessed.jsonl

Each output record contains:
  region         | "Caribbean" | "UK" | "Canada" | "International" | "Unknown"
  job_category   | "AI" | "CS" | "IT" | "Unrelated" | "Unknown"
  source         | "linkedin" | "workopolis" | "reed"
  search_term    | original search/query term used to find the job
  job_title      | job title as listed
  url            | source URL (None for LinkedIn records without one)
  clean_text     | SkillNER-friendly cleaned + tokenized text

Cleaning steps (applied to all sources):
  1. Strip HTML tags
  2. Fix letter-spaced words (e.g. "c o d e" -> "code")
  3. Normalize tech variants to canonical SkillNER surface forms
  4. Remove characters outside [a-zA-Z0-9 + # . / - _]
  5. Normalize whitespace
  6. Tokenize with spaCy (remove stopwords, punctuation, single chars)

Deduplication (applied globally across all sources):
  - Layer 1: URL deduplication
  - Layer 2: Content hash (MD5) deduplication
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from bs4 import BeautifulSoup
import spacy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "all_jobs_preprocessed.jsonl"

LINKEDIN_FILE   = DATA_DIR / "linkedin_jobs.jsonl"
WORKOPOLIS_FILE = DATA_DIR / "workopolis_jobs.json"
REED_FILE       = DATA_DIR / "reed_jobs.json"

# ---------------------------------------------------------------------------
# Role lists for category inference (used for LinkedIn + fallback)
# ---------------------------------------------------------------------------

CS_ROLES = [
    "software engineer", "software developer", "backend developer",
    "frontend developer", "full stack developer", "mobile application developer",
    "game developer", "embedded systems engineer", "devops engineer",
    "cloud engineer",
]

IT_ROLES = [
    "it support specialist", "help desk technician", "network administrator",
    "systems administrator", "cybersecurity analyst", "information security analyst",
    "database administrator", "it project manager", "it operations analyst",
    "infrastructure engineer",
]

AI_ROLES = [
    "data scientist", "machine learning engineer", "ai engineer",
    "data analyst", "business intelligence analyst", "nlp engineer",
    "computer vision engineer", "data engineer", "mlops engineer",
    "applied scientist",
]

UNRELATED_ROLES = [
    "registered nurse", "primary school teacher", "accountant",
    "human resources officer", "sales representative", "restaurant manager",
    "chef", "warehouse supervisor", "construction supervisor", "pharmacist",
]

# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

def detect_region_from_url(url: Optional[str]) -> str:
    if not url:
        return "Unknown"
    url = url.lower()
    if "caribbeanjobs.com" in url:
        return "Caribbean"
    if "reed.co.uk" in url:
        return "UK"
    if "workopolis.com" in url:
        return "Canada"
    if "linkedin.com" in url:
        return "International"
    return "Unknown"


def detect_region_from_record(rec: Dict[str, Any], source: str) -> str:
    """
    For LinkedIn records the region field is already set by the scraper.
    For other sources, infer from the URL.
    """
    if source == "linkedin":
        return rec.get("region") or detect_region_from_url(rec.get("url"))
    return detect_region_from_url(rec.get("url"))


# Only process records that match this target role
TARGET_ROLE = "data scientist"


def is_target_role(job_title: Optional[str], search_term: Optional[str]) -> bool:
    jt = (job_title or "").lower()
    st = (search_term or "").lower()
    return "data scientist" in jt or "data scientist" in st


# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

def infer_job_category(job_title: str, search_term: str, raw_category: Optional[str] = None) -> str:
    """
    Priority order:
      1. Explicit category field from scraper (Workopolis / Reed sometimes supply this)
      2. Match against role lists using title + search term
    """
    if raw_category:
        cat = raw_category.strip().upper()
        if cat in {"AI", "CS", "IT", "UNRELATED"}:
            return cat

    candidates = [s.lower() for s in [job_title or "", search_term or ""] if s]

    for role in AI_ROLES:
        if any(role in c for c in candidates):
            return "AI"
    for role in CS_ROLES:
        if any(role in c for c in candidates):
            return "CS"
    for role in IT_ROLES:
        if any(role in c for c in candidates):
            return "IT"
    for role in UNRELATED_ROLES:
        if any(role in c for c in candidates):
            return "Unrelated"
    return "Unknown"


# ---------------------------------------------------------------------------
# Text normalisation
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
# Source loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping bad JSON at {path.name}:{lineno}")


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
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


def iter_linkedin(path: Path) -> Iterator[Dict[str, Any]]:
    """Yields normalised record dicts from the LinkedIn JSONL file."""
    for rec in _load_jsonl(path):
        yield {
            "source":       "linkedin",
            "region_raw":   rec.get("region"),
            "raw_category": None,                          # inferred from title
            "search_term":  rec.get("title_query"),
            "job_title":    rec.get("job_title", ""),
            "url":          rec.get("url"),
            "description":  rec.get("description", ""),
            "_rec":         rec,                           # keep original for region helper
        }


def iter_workopolis(path: Path) -> Iterator[Dict[str, Any]]:
    for rec in _load_json_array(path):
        yield {
            "source":       "workopolis",
            "region_raw":   None,
            "raw_category": rec.get("category"),
            "search_term":  rec.get("search_term") or rec.get("searched_role"),
            "job_title":    rec.get("title") or rec.get("job_title", ""),
            "url":          rec.get("url"),
            "description":  rec.get("description", ""),
            "_rec":         rec,
        }


def iter_reed(path: Path) -> Iterator[Dict[str, Any]]:
    for rec in _load_json_array(path):
        yield {
            "source":       "reed",
            "region_raw":   None,
            "raw_category": rec.get("category"),
            "search_term":  rec.get("search_term") or rec.get("searched_role"),
            "job_title":    rec.get("title") or rec.get("job_title", ""),
            "url":          rec.get("url"),
            "description":  rec.get("description", ""),
            "_rec":         rec,
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        (LINKEDIN_FILE,   iter_linkedin,   "LinkedIn"),
        (WORKOPOLIS_FILE, iter_workopolis, "Workopolis"),
        (REED_FILE,       iter_reed,       "Reed UK"),
    ]

    seen_urls:   set = set()
    seen_hashes: set = set()

    total_raw  = 0
    total_out  = 0
    skipped_missing = 0
    skipped_url_dup  = 0
    skipped_hash_dup = 0
    skipped_non_target = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:

        for file_path, loader_fn, label in sources:

            if not file_path.exists():
                print(f"[SKIP] {label} file not found: {file_path.resolve()}")
                continue

            print(f"[INFO] Loading {label} ({file_path.name})...")
            source_count = 0

            for norm in loader_fn(file_path):
                total_raw += 1
                # Filter to target role only
                if not is_target_role(norm.get("job_title"), norm.get("search_term")):
                    skipped_non_target += 1
                    continue
                source = norm["source"]

                # --- URL dedup ---
                url = norm.get("url")
                if url:
                    if url in seen_urls:
                        skipped_url_dup += 1
                        continue
                    seen_urls.add(url)

                # --- Must have a description ---
                raw_desc = norm.get("description", "").strip()
                if not raw_desc:
                    skipped_missing += 1
                    continue

                # --- Clean + tokenize ---
                cleaned    = clean_text(raw_desc)
                tokens     = tokenize(cleaned)
                if not tokens:
                    skipped_missing += 1
                    continue

                clean_text_str = " ".join(tokens)

                # --- Content hash dedup ---
                content_hash = hashlib.md5(clean_text_str.encode()).hexdigest()
                if content_hash in seen_hashes:
                    skipped_hash_dup += 1
                    continue
                seen_hashes.add(content_hash)

                # --- Resolve fields ---
                job_title   = norm.get("job_title", "")
                search_term = norm.get("search_term", "")
                region      = detect_region_from_record(norm["_rec"], source)
                category    = infer_job_category(
                    job_title   = job_title,
                    search_term = search_term,
                    raw_category= norm.get("raw_category"),
                )

                record = {
                    "source":       source,
                    "region":       region,
                    "job_category": category,
                    "search_term":  search_term or None,
                    "job_title":    job_title or None,
                    "url":          url,
                    "clean_text":   clean_text_str,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_out  += 1
                source_count += 1

                if total_out % 200 == 0:
                    print(f"[INFO] Written {total_out} records so far...")

            print(f"[INFO] {label}: contributed {source_count} records.")

    print()
    print("=" * 50)
    print(f"  Raw records seen      : {total_raw}")
    print(f"  Written to output     : {total_out}")
    print(f"  Skipped (URL dup)     : {skipped_url_dup}")
    print(f"  Skipped (content dup) : {skipped_hash_dup}")
    print(f"  Skipped (empty/bad)   : {skipped_missing}")
    print(f"  Skipped (non-target)  : {skipped_non_target}")
    print(f"  Output file           : {OUTPUT_FILE.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()