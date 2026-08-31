#!/usr/bin/env python3
"""
mooc_skill_extractor.py — Extract skills from scraped Coursera course data.

Input  : data/processed/coursera_courses_preprocessed.jsonl
Output : data/skills/coursera_courses_skills.jsonl

For each course, combines the description and syllabus into a single text
and runs SkillNER to extract skills. Output preserves all original fields
and adds a `skills` list.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import spacy
from spacy.matcher import PhraseMatcher

from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor


# ==============================
# CONFIG
# ==============================

INPUT_FILE  = Path("data/processed/coursera_courses_preprocessed.jsonl")
OUTPUT_FILE = Path("data/skills/coursera_courses_skills.jsonl")

SPACY_MODEL          = "en_core_web_sm"
NGRAM_SCORE_THRESHOLD = 0.55

VALID_SINGLE_TOKENS = {"c", "r", "ai", "ml"}

GENERIC_PHRASES = {
    "show less", "show more", "applied learning project",
    "community", "reviews", "overview",
    # Generic single words
    "case", "reading", "readings", "recall", "commenting", "draft",
    "planning", "claim", "onboarding", "scale",
    "logging", "mapping", "profiling", "transformation", "operations", "innovation", "resilience", "treasury",
    "workflows", "infrastructure", "management", "collaboration",
    "empathy", "sales", "storytelling", "presentations", "scheduling",
    "sorting", "branding", "stewardship", "business",
    "simulations", "trajectory", "framing", "leadership",
}

# Exact noisy phrases to remove
NOISE_EXACT = {
    # Duplicated/concatenated noise
    "governance governance", "tests tests", "course course",
    "data engineers data", "data software", "managers database",
    "engineers computer", "engineers design", "engineers professionals",
    "data intelligent", "code generative", "machine learn",
    # Awkward verb phrases
    "study layouts", "root cause actions", "effectively market",
    "design build", "structured business", "writing persuasive",
    "manage demand", "build manage", "managing controlling",
    "solve complex problems", "solving problems", "complex problems",
    "optimizing software", "performance enhancing", "sort algorithm",
    # Noise concatenations
    "practical industry", "engineering landscape", "banking investments",
    "portfolio projects", "projects global", "industry landscape",
    "tools help", "applications enterprise", "architecture data",
    "developers students", "risks financial", "operational cyber",
    "vulnerabilities assess", "physics computational",
    "administrators business", "professional project manager",
    "service provider", "cloud architecture", "experiential learning",
    "readings guided", "visualization communicate", "design driven",
    "present data", "marketing business", "skills influence",
    "statistics business", "learning outcomes", "point sale",
}

GENERIC_WORDS = {
    "introduction", "concept", "concepts", "topic", "topics",
    "area", "areas", "aspect", "aspects", "system", "systems",
    "study", "studies", "method", "methods", "use", "uses",
}


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
# SKILL CLEANING
# ==============================

def normalize_skill(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    # Fix datum -> data variants
    datum_fixes = {
        "clean datum":        "data cleaning",
        "categorical datum":  "categorical data",
        "source datum":       "data sources",
        "unstructured datum": "unstructured data",
        "ingested data":      "data ingestion",
        "intelligent data":   "data intelligence",
        "extract transform load": "ETL",
        "machine learn":      "machine learning",
        "discrete mathematic": "discrete mathematics",
        "problem solve":      "problem solving",
        "fluid dynamic":      "fluid dynamics",
        "computational fluid dynamic": "computational fluid dynamics",
        "digital forensic":   "digital forensics",
    }
    return datum_fixes.get(text, text)


def is_valid_skill(skill: str) -> bool:
    s = skill.strip().lower()
    if not s:
        return False
    if s in VALID_SINGLE_TOKENS:
        return True
    if s in GENERIC_PHRASES:
        return False
    if s in NOISE_EXACT:
        return False
    if len(s) == 1:
        return False
    if not any(c.isalpha() for c in s):
        return False

    parts = s.split()

    # Too long to be a meaningful skill
    if len(parts) > 4:
        return False

    # All generic filler words
    if all(p in GENERIC_WORDS for p in parts):
        return False

    # Duplicated word (e.g. "tests tests", "course course")
    if len(parts) == 2 and parts[0] == parts[1]:
        return False

    # Verb-led awkward phrases (e.g. "collect data" is fine but "managing controlling" is not)
    BAD_VERB_PAIRS = {
        ("managing", "controlling"), ("solving", "problems"),
        ("optimizing", "software"), ("effectively", "market"),
    }
    if len(parts) == 2 and tuple(parts) in BAD_VERB_PAIRS:
        return False

    return True


def collect_skills(annotations: Dict[str, Any]) -> List[str]:
    results      = annotations.get("results", {}) if isinstance(annotations, dict) else {}
    full_matches = results.get("full_matches", []) or []
    ngram_scored = results.get("ngram_scored", []) or []

    best: Dict[str, float] = {}

    def ingest(items: List[Dict[str, Any]], source: str) -> None:
        for it in items:
            raw = (it.get("doc_node_value") or "").strip()
            if not raw:
                continue
            score = float(it.get("score", 0.0) or 0.0)
            if source == "ngram" and score < NGRAM_SCORE_THRESHOLD:
                continue
            skill = normalize_skill(raw)
            if not is_valid_skill(skill):
                continue
            if skill not in best or score > best[skill]:
                best[skill] = score

    ingest(full_matches, "full")
    ingest(ngram_scored, "ngram")

    sorted_skills: List[Tuple[str, float]] = sorted(
        best.items(), key=lambda x: (-x[1], x[0])
    )
    return [s for s, _ in sorted_skills]


# ==============================
# MAIN
# ==============================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE.resolve()}")

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        raise SystemExit(f"Run: python -m spacy download {SPACY_MODEL}")

    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

    records = []
    failed  = 0

    print("[INFO] Extracting skills from Coursera courses...")

    for rec in iter_jsonl(INPUT_FILE):
        topic_id    = rec.get("topic_id")
        topic_label = rec.get("topic_label", "")
        course_title = rec.get("course_title", "")

        text = (rec.get("clean_text") or "").strip()

        if not text:
            print(f"  [WARN] Topic {topic_id} — no text to extract from")
            skills = []
        else:
            try:
                annotations = skill_extractor.annotate(text)
                skills      = collect_skills(annotations)
            except Exception as e:
                print(f"  [WARN] Topic {topic_id} extraction failed: {e}")
                skills = []
                failed += 1

        print(f"  Topic {str(topic_id):<3} | {course_title[:50]:<50} | {len(skills)} skills")

        records.append({
            "topic_id":    topic_id,
            "topic_label": topic_label,
            "course_title": course_title,
            "provider":    rec.get("provider", ""),
            "skills":      skills,
        })

    write_jsonl(OUTPUT_FILE, records)

    print()
    print("=" * 60)
    print(f"  Courses processed  : {len(records)}")
    print(f"  Extraction failures: {failed}")
    print(f"  Output             : {OUTPUT_FILE.resolve()}")
    print()

    # Print skills per course for quick review
    print("  Skills per course:")
    for r in records:
        print(f"    Topic {r['topic_id']}: {r['skills'][:8]}")
    print("=" * 60)


if __name__ == "__main__":
    main()