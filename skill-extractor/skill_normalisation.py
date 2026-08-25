#!/usr/bin/env python3
"""
Shared skill normalisation for the course and job extractors.

Both extractors previously normalised skills independently, which let the two
corpora drift apart: the same concept surfaced as "code reviews" in job ads and
"code review" in syllabi, or "data modeling" in one and "data modelling" in the
other. Because the gap analysis matches skills as exact string tokens, those
pairs never matched and the curriculum received no credit for skills it teaches.

This module is the single source of truth for:

  1. British spelling      -- "visualization" -> "visualisation"
  2. Singular forms        -- "code reviews"  -> "code review"
  3. Junk removal          -- job-ad boilerplate, SkillNER n-gram scrambles and
                              contentless generic words

Import it from both extractors so the two corpora stay in lockstep.
"""

import re
from typing import List, Tuple, Callable, Union


# ==============================
# BRITISH SPELLING
# ==============================

_UK_SPELLING: List[Tuple[str, Union[str, Callable]]] = [
    (r"(\w*?)izations?\b",        lambda m: m.group(1) + "isation"),
    (r"(\w*?)iz(e|es|ed|ing)\b",  lambda m: m.group(1) + "is" + m.group(2)[1:]),
    (r"(\w*?)yz(e|es|ed|ing)\b",  lambda m: m.group(1) + "ys" + m.group(2)[1:]),
    (r"\b(model|label|travel|cancel)(ing|ed)\b", r"\1l\2"),
    (r"\bcolor",     "colour"),
    (r"\bbehavior",  "behaviour"),
    (r"\bcatalog\b", "catalogue"),
    (r"\bdefense\b", "defence"),
    (r"\bcenter",    "centre"),
]


# ==============================
# SINGULARISATION
# ==============================

# Words that end in "s" but are already singular, or whose singular form would
# be wrong. Stripping the "s" from these produces nonsense ("statistics" ->
# "statistic", "aws" -> "aw"), so they are left alone.
_PROTECTED_PLURALS = {
    "statistics", "mathematics", "analytics", "economics", "physics", "ethics",
    "logistics", "genetics", "robotics", "graphics", "dynamics", "mechanics",
    "semantics", "informatics", "numerics", "aerodynamics",
    "series", "news", "access", "sales", "operations", "communications",
    "analysis", "status", "bus", "gas", "lens", "campus", "focus",

    # Technology names that end in "s". Singularising these produces nonsense
    # ("pandas" -> "panda", "kerberos" -> "kerbero"), so they are pinned.
    "aws", "devops", "ops", "kubernetes", "jenkins", "sass",
    "pandas", "databricks", "kerberos", "quickbooks", "redis", "rails",
    "windows", "macos", "docs", "analytics platforms",
}

# Irregular plurals the rules below cannot derive.
_IRREGULAR_SINGULARS = {
    "analyses":  "analysis",
    "matrices":  "matrix",
    "indices":   "index",
    "vertices":  "vertex",
    "criteria":  "criterion",
    "schemata":  "schema",
    "viruses":   "virus",
    "buses":     "bus",
    "lenses":    "lens",
}


def _apply_uk_spelling(text: str) -> str:
    for pattern, replacement in _UK_SPELLING:
        text = re.sub(pattern, replacement, text)
    return text


def _singularise_last_word(text: str) -> str:
    """Singularise only the final word of a phrase.

    Skills are head-final ("code reviews", "design patterns"), so the last word
    carries the plural. Earlier words are left untouched.
    """
    parts = text.split()
    if not parts:
        return text

    last = parts[-1]

    if last in _IRREGULAR_SINGULARS:
        parts[-1] = _IRREGULAR_SINGULARS[last]
        return " ".join(parts)

    if last in _PROTECTED_PLURALS:
        return text

    if re.search(r"(sses|ches|shes|xes|zes)$", last):
        # "processes" -> "process", "classes" -> "class"
        last = last[:-2]
    elif re.search(r"[^aeiou]ies$", last) and len(last) > 4:
        # "technologies" -> "technology"
        last = last[:-3] + "y"
    elif re.search(r"(ss|us|is|ics|ous)$", last):
        # already singular, or not a plural at all
        pass
    elif last.endswith("s") and len(last) > 4:
        last = last[:-1]

    parts[-1] = last
    return " ".join(parts)


def canonicalise_skill(text: str) -> str:
    """Normalise a skill string to its canonical form.

    Lowercases, collapses whitespace, converts to British spelling and
    singularises the head word. Safe to call more than once.
    """
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = _apply_uk_spelling(text)
    text = _singularise_last_word(text)
    return text


# ==============================
# JUNK REMOVAL
# ==============================

# Benefits, equal-opportunity statements and other job-ad boilerplate that
# SkillNER mistook for skills.
_JOB_AD_BOILERPLATE = {
    "dental care", "health dental", "dental", "vision dental",
    "nice",                      # from "nice to have"
    "disabilities", "disability",
    "mental health", "development budget", "budget",
    "safe", "latin", "english", "legislation", "onboarding",
    "paid leave", "sick leave", "equal opportunity",
}

# SkillNER sliding-window collisions: fragments of real phrases with the words
# reversed or split. The correctly-ordered form is usually captured separately,
# so these are duplicates as well as noise.
_NGRAM_SCRAMBLES = {
    "analysis data", "quality data", "modeling data", "modelling data",
    "applications data", "analysis decision", "analysis systems",
    "analysis needs", "statistics computer", "statistics economics",
    "mathematics statistics", "mathematics computer",
    "engineers data", "engineering data", "engineers product",
    "engineering product", "data engineers",
    "management communication", "collaboration communication",
    "partner business", "processes developing", "designing building",
    "support development", "service customers", "service delivery",
    "based models", "based systems", "solution design",
}

# Single words with no skill content. Several are false matches on common verbs
# -- "act", "acting" and "react" appear together, which indicates SkillNER
# matched the verb rather than the React framework.
_GENERIC_NON_SKILLS = {
    "track", "tracking", "source", "act", "acting", "react",
    "read", "reach", "case", "fix", "massive", "agenda", "scheme",
    "framing", "templates", "template", "logging", "reduction",
    "primer", "underscores", "perspective", "blend learning",
}

JUNK_SKILLS = _JOB_AD_BOILERPLATE | _NGRAM_SCRAMBLES | _GENERIC_NON_SKILLS

# Compare against canonical forms so "disabilities" is caught by "disability"
# and "code reviews" would be caught by "code review".
_CANONICAL_JUNK = {canonicalise_skill(s) for s in JUNK_SKILLS}


def is_junk_skill(text: str) -> bool:
    """True if the skill is boilerplate, an n-gram artefact or contentless."""
    return canonicalise_skill(text) in _CANONICAL_JUNK
