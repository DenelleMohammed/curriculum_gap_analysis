#!/usr/bin/env python3
"""
coursera_scraper.py — Selenium-based Coursera scraper.

For each DS curriculum gap topic:
  1. Searches Coursera for the top course
    2. Records the top course title and URL

Output:
  data/mooc/coursera_gap_courses.jsonl
  data/mooc/coursera_gap_courses.csv

Requirements:
  pip install selenium beautifulsoup4
  ChromeDriver matching your installed Chrome version must be on PATH.
  Download from: https://googlechromelabs.github.io/chrome-for-testing/
"""

import csv
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


# ==============================
# CONFIG
# ==============================

OUTPUT_DIR   = Path("data/mooc")
OUTPUT_JSONL = OUTPUT_DIR / "coursera_gap_courses.jsonl"
OUTPUT_CSV   = OUTPUT_DIR / "coursera_gap_courses.csv"

HEADLESS       = True   # Set False to see the browser / debug
PAGE_LOAD_WAIT = 12     # seconds to wait for page elements


# ==============================
# GAP TOPICS
# ==============================

GAP_TOPICS = [
    {"topic_id": 14, "label": "databases embedding, knowledge graph, rest apis", "query": "databases embedding, knowledge graph, rest apis"},
    {"topic_id": 6,  "label": "economic indicator, requisition, systems thinking", "query": "economic indicator, requisition, systems thinking"},
    {"topic_id": 0,  "label": "refinement, data exploration, business analytics", "query": "refinement, data exploration, business analytics"},
    {"topic_id": 2,  "label": "analytical skill, projects collaborate, c", "query": "analytical skill, projects collaborate, c"},
    {"topic_id": 12, "label": "business acumen, testing regression, google cloud", "query": "business acumen, testing regression, google cloud"},
    {"topic_id": 7,  "label": "analysis business, benchmarking, strategies business", "query": "analysis business, benchmarking, strategies business"},
    {"topic_id": 5,  "label": "sprint planning, tested code, requirements manage", "query": "sprint planning, tested code, requirements manage"},
]


# ==============================
# DRIVER
# ==============================

def build_driver() -> webdriver.Chrome:
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def scroll_page(driver: webdriver.Chrome) -> None:
    """Scroll slowly to trigger lazy-loaded content."""
    total = driver.execute_script("return document.body.scrollHeight")
    for pos in range(0, total, 400):
        driver.execute_script(f"window.scrollTo(0, {pos});")
        time.sleep(0.15)
    time.sleep(SCROLL_PAUSE)


# ==============================
# SEARCH
# ==============================

def search_top_course(driver: webdriver.Chrome, query: str) -> Optional[Dict[str, str]]:
    url = f"https://www.coursera.org/search?query={quote_plus(query)}"
    try:
        driver.get(url)
        WebDriverWait(driver, PAGE_LOAD_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR,
                "a[href*='/learn/'], a[href*='/specializations/'], a[href*='/professional-certificates/']"
            ))
        )
        time.sleep(random.uniform(2, 4))
        soup = BeautifulSoup(driver.page_source, "html.parser")

        for card_sel in ["li.cds-9", "[data-e2e='search-result-card']", "div.css-1qchnft"]:
            cards = soup.select(card_sel)
            if cards:
                card     = cards[0]
                title_el = card.select_one("h3, h2")
                link_el  = card.select_one(
                    "a[href*='/learn/'], a[href*='/specializations/'], a[href*='/professional-certificates/']"
                )
                if title_el and link_el:
                    href = link_el.get("href", "")
                    if not href.startswith("http"):
                        href = "https://www.coursera.org" + href
                    return {
                        "title": title_el.get_text(strip=True),
                        "url": href,
                    }

        # Fallback
        links = soup.select(
            "a[href*='/learn/'], a[href*='/specializations/'], a[href*='/professional-certificates/']"
        )
        if links:
            href = links[0].get("href", "")
            if not href.startswith("http"):
                href = "https://www.coursera.org" + href
            return {"title": links[0].get_text(strip=True) or query, "url": href}

    except TimeoutException:
        print("  [WARN] Search timed out")
    except WebDriverException as e:
        print(f"  [ERROR] Search failed: {e}")
    return None


# ==============================
# IO
# ==============================

def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    flat = []
    for rec in records:
        flat.append(rec)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)


# ==============================
# MAIN
# ==============================

def main():
    results = []
    driver  = build_driver()

    try:
        for gap in GAP_TOPICS:
            print(f"\n[Topic {gap['topic_id']}] {gap['label']}")
            print(f"  Query : {gap['query']}")

            course = search_top_course(driver, gap["query"])

            if not course:
                print("  [WARN] No course found.")
                results.append({
                    "topic_id": gap["topic_id"], "topic_label": gap["label"],
                    "search_query": gap["query"], "course_title": "Not found",
                    "course_url": "",
                })
                continue

            print(f"  Found : {course['title']}")
            print(f"  URL   : {course['url']}")

            results.append({
                "topic_id":      gap["topic_id"],
                "topic_label":   gap["label"],
                "search_query":  gap["query"],
                "course_title":  course["title"],
                "course_url":    course["url"],
            })

            time.sleep(random.uniform(4, 8))

    finally:
        driver.quit()

    write_jsonl(OUTPUT_JSONL, results)
    write_csv(OUTPUT_CSV, results)

    print()
    print("=" * 60)
    print(f"  Gaps processed : {len(results)}")
    print(f"  JSONL          : {OUTPUT_JSONL.resolve()}")
    print(f"  CSV            : {OUTPUT_CSV.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()