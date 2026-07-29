#!/usr/bin/env python3
"""
course_skill_extractor.py — SkillNER-based skill extractor for all courses.

Input  : data/processed/all_courses_preprocessed.jsonl
Output :
  - data/skills/all_courses_skills.jsonl         (all courses)
  - data/skills/cs_courses_skills.jsonl          (CS thematic area only)
  - data/skills/it_courses_skills.jsonl          (IT thematic area only)
  - data/skills/cs_special_course_skills.jsonl   (B.Sc. Computer Science
                                                   Special degree courses only,
                                                   core + elective)

Each output record contains:
  course_code     | e.g. "COMP 3605"
  course_name     | full course name
  thematic_areas  | list of thematic area tags
  skills          | list of extracted, filtered skills
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

INPUT_FILE     = Path("data/processed/all_courses_preprocessed.jsonl")
OUTPUT_ALL     = Path("data/skills/all_courses_skills.jsonl")
OUTPUT_CS      = Path("data/skills/cs_courses_skills.jsonl")
OUTPUT_IT      = Path("data/skills/it_courses_skills.jsonl")
OUTPUT_SPECIAL = Path("data/skills/cs_special_course_skills.jsonl")

SPACY_MODEL          = "en_core_web_sm"
NGRAM_SCORE_THRESHOLD = 0.55
MAX_SKILL_TOKENS      = 6

# B.Sc. Computer Science (Special) course list — core + elective
CS_SPECIAL_COURSE_CODES = {
    # Core — Level I
    "COMP 1600", "COMP 1601", "COMP 1602", "COMP 1603", "COMP 1604",
    "INFO 1600", "INFO 1601", "MATH 1115", "FOUN 1101", "FOUN 1105",
    # Core — Level II/III
    "COMP 2601", "COMP 2602", "COMP 2603", "COMP 2604", "COMP 2605",
    "COMP 2606", "COMP 2611", "COMP 3601", "COMP 3602", "COMP 3603",
    "COMP 3613", "INFO 2602", "INFO 2604", "INFO 3604", "MATH 2250",
    "FOUN 1301",
    # Elective — Level II/III
    "COMP 3605", "COMP 3606", "COMP 3607", "COMP 3608", "COMP 3609",
    "COMP 3610", "COMP 3611", "COMP 3612", "INFO 2605", "INFO 3600",
    "INFO 3605", "INFO 3606", "INFO 3607", "INFO 3608", "INFO 3609",
    "INFO 3610", "INFO 3611",
}

# Short tokens that are valid skills
VALID_SINGLE_TOKENS = {"c", "r"}

# Valid e-commerce style phrases
VALID_E_PHRASES = {"e business", "e commerce"}

# Junk single-character tokens
BAD_SINGLE_TOKENS = {
    "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "at",
    "e", "b", "d", "x", "y", "z"
}

# Generic filler words that make noisy phrases
GENERIC_WORDS = {
    "introduction", "concept", "concepts", "topic", "topics",
    "area", "areas", "aspect", "aspects", "system", "systems",
    "study", "studies", "method", "methods", "use", "uses",
    "case", "cases"
}

# Canonical replacements for common variants
REPLACEMENTS = {
    "c programming": "c",
    "r language":    "r",
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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ==============================
# NORMALISATION
# ==============================

def canonicalize_symbolic_skills(text: str) -> str:
    text = re.sub(r"\bc\s*\+\s*\+(?=\s|$)", "c++", text)
    return text


def normalize_skill(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = canonicalize_symbolic_skills(text)
    return REPLACEMENTS.get(text, text)


# ==============================
# NOISE FILTERS
# ==============================

def is_repeated_phrase(skill: str) -> bool:
    parts = skill.split()
    return len(parts) >= 2 and len(set(parts)) == 1


def has_too_many_short_tokens(skill: str) -> bool:
    parts = skill.split()
    if not parts:
        return True
    if len(parts) == 1:
        return parts[0] not in VALID_SINGLE_TOKENS and len(parts[0]) == 1
    short_count = sum(
        1 for p in parts if len(p) == 1 and p not in VALID_SINGLE_TOKENS
    )
    return short_count >= 2


def starts_or_ends_with_bad_short_token(skill: str) -> bool:
    if skill in VALID_E_PHRASES:
        return False
    parts = skill.split()
    if not parts:
        return True
    first, last = parts[0], parts[-1]
    if len(first) == 1 and first not in VALID_SINGLE_TOKENS:
        return True
    if len(last) == 1 and last not in VALID_SINGLE_TOKENS:
        return True
    return False


def looks_like_noise(skill: str) -> bool:
    if not skill:
        return True
    parts = skill.split()

    if len(parts) > MAX_SKILL_TOKENS:
        return True
    if len(skill) == 1 and skill not in VALID_SINGLE_TOKENS:
        return True
    if skill in BAD_SINGLE_TOKENS:
        return True
    if is_repeated_phrase(skill):
        return True
    if has_too_many_short_tokens(skill):
        return True
    if len(parts) > 1 and starts_or_ends_with_bad_short_token(skill):
        return True
    if all(p in GENERIC_WORDS for p in parts):
        return True

    # Reject anything with characters outside a-z, 0-9, + # . - /
    compact = re.sub(r"[a-z0-9+#.\-/ ]", "", skill)
    if compact:
        return True

    return False


def should_keep_match(name: str, score: float, source: str) -> bool:
    skill = normalize_skill(name)
    if looks_like_noise(skill):
        return False
    if source == "full":
        return True
    if source == "ngram":
        if score < NGRAM_SCORE_THRESHOLD:
            return False
        parts = skill.split()
        if len(parts) == 1 and skill in GENERIC_WORDS:
            return False
        return True
    return False


# ==============================
# SKILL COLLECTION
# ==============================

def collect_skills(annotations: Dict[str, Any]) -> List[str]:
    results      = annotations.get("results", {}) if isinstance(annotations, dict) else {}
    full_matches = results.get("full_matches", []) or []
    ngram_scored = results.get("ngram_scored", []) or []

    best: Dict[str, float] = {}

    def ingest(items: List[Dict[str, Any]], source: str) -> None:
        for it in items:
            raw_name = (it.get("doc_node_value") or "").strip()
            if not raw_name:
                continue
            score = float(it.get("score", 0.0) or 0.0)
            skill = normalize_skill(raw_name)
            if not should_keep_match(skill, score, source):
                continue
            if skill not in best or score > best[skill]:
                best[skill] = score

    ingest(full_matches, "full")
    ingest(ngram_scored, "ngram")

    sorted_items: List[Tuple[str, float]] = sorted(
        best.items(), key=lambda x: (-x[1], x[0])
    )
    return [name for name, _ in sorted_items]


# ==============================
# MAIN
# ==============================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        raise SystemExit(
            f"spaCy model '{SPACY_MODEL}' not found.\n"
            f"Run: python -m spacy download {SPACY_MODEL}"
        )

    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

    rows: List[Dict[str, Any]] = []
    failed = 0

    print("[INFO] Extracting skills from DCIT courses...")

    for rec in iter_jsonl(INPUT_FILE):
        course_code = (rec.get("course_code") or "").strip()
        course_name = (
            rec.get("course_name")
            or rec.get("course_title")
            or rec.get("name")
            or ""
        ).strip()
        if not course_name and course_code:
            course_name = course_code

        # preprocessor stores cleaned text under "skills" key
        text = (rec.get("skills") or rec.get("clean_text") or "").strip()

        if not course_code:
            continue

        # thematic area inferred from course code prefix
        if course_code.startswith("COMP"):
            thematic_areas = ["computer_science"]
        elif course_code.startswith("INFO"):
            thematic_areas = ["information_technology"]
        else:
            thematic_areas = []

        # SkillNER works better without underscores
        text = text.replace("_", " ")

        if not text:
            rows.append({
                "course_code":   course_code,
                "course_name":   course_name,
                "thematic_areas": thematic_areas,
                "skills":        [],
            })
            continue

        try:
            annotations = skill_extractor.annotate(text)
            skills      = collect_skills(annotations)
        except Exception as e:
            print(f"[WARN] Failed on {course_code}: {e}")
            skills = []
            failed += 1

        rows.append({
            "course_code":    course_code,
            "course_name":    course_name,
            "thematic_areas": thematic_areas,
            "skills":         skills,
        })

        print(f"  {course_code:<12} → {len(skills)} skills")

    # Split by thematic area
    cs_rows = [r for r in rows if "computer_science"      in r.get("thematic_areas", [])]
    it_rows = [r for r in rows if "information_technology" in r.get("thematic_areas", [])]

    # Filter to B.Sc. Computer Science (Special) degree course list
    special_rows = [r for r in rows if r.get("course_code") in CS_SPECIAL_COURSE_CODES]

    write_jsonl(OUTPUT_ALL,     rows)
    write_jsonl(OUTPUT_CS,      cs_rows)
    write_jsonl(OUTPUT_IT,      it_rows)
    write_jsonl(OUTPUT_SPECIAL, special_rows)

    print()
    print("=" * 50)
    print(f"  Courses processed     : {len(rows)}")
    print(f"  Extraction failures   : {failed}")
    print(f"  CS courses            : {len(cs_rows)}")
    print(f"  IT courses            : {len(it_rows)}")
    print(f"  CS Special courses    : {len(special_rows)} / {len(CS_SPECIAL_COURSE_CODES)}")
    print()
    print(f"  Output (all)          : {OUTPUT_ALL.resolve()}")
    print(f"  Output (CS)           : {OUTPUT_CS.resolve()}")
    print(f"  Output (IT)           : {OUTPUT_IT.resolve()}")
    print(f"  Output (CS Special)   : {OUTPUT_SPECIAL.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()