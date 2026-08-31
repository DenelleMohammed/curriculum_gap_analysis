#!/usr/bin/env python3
"""
coursera_scraper.py — Selenium-based Coursera scraper.

For each DS curriculum gap topic:
  1. Searches Coursera for the top course
  2. Visits the course page and waits for JavaScript to render
  3. Extracts: description and syllabus modules

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
SCROLL_PAUSE   = 2.0    # seconds after scrolling to let content load


# ==============================
# GAP TOPICS
# ==============================

GAP_TOPICS = [
    {"topic_id": 9,  "label": "mysql, acting, feature engineering",            "query": "mysql feature engineering"},
    {"topic_id": 4,  "label": "engineers data, airflow, statistics computer",  "query": "engineers data airflow statistics computer"},
    {"topic_id": 5,  "label": "latin, financial institutions, mentorship",      "query": "latin financial institutions mentorship"},
    {"topic_id": 1,  "label": "cloud computing, framing, finance",             "query": "cloud computing framing finance"},
    {"topic_id": 14, "label": "customer experiences, cyber security, business cases", "query": "customer experiences cyber security business cases"},
    {"topic_id": 11, "label": "hospitality, demand forecasting, library",      "query": "hospitality demand forecasting library"},
    {"topic_id": 10, "label": "data governance, customer service, detail oriented", "query": "data governance customer service detail oriented"},
    {"topic_id": 2,  "label": "executive, statistics computer, data collection", "query": "executive statistics computer data collection"},
    {"topic_id": 3,  "label": "mathematics statistics, embedding, solution design", "query": "mathematics statistics embedding solution design"},
    {"topic_id": 7,  "label": "data collection, executive, crm",               "query": "data collection executive crm"},
    {"topic_id": 6,  "label": "quality data, business operations, data extraction", "query": "quality data business operations data extraction"},
    {"topic_id": 12, "label": "case, framing, testing data",                   "query": "case framing testing data"},
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
                    prov = card.select_one("p")
                    return {
                        "title":    title_el.get_text(strip=True),
                        "url":      href,
                        "provider": prov.get_text(strip=True) if prov else "",
                    }

        # Fallback
        links = soup.select(
            "a[href*='/learn/'], a[href*='/specializations/'], a[href*='/professional-certificates/']"
        )
        if links:
            href = links[0].get("href", "")
            if not href.startswith("http"):
                href = "https://www.coursera.org" + href
            return {"title": links[0].get_text(strip=True) or query, "url": href, "provider": ""}

    except TimeoutException:
        print("  [WARN] Search timed out")
    except WebDriverException as e:
        print(f"  [ERROR] Search failed: {e}")
    return None


# ==============================
# COURSE PAGE
# ==============================

def expand_accordions(driver: webdriver.Chrome) -> None:
    """Click all collapsed syllabus accordion buttons."""
    try:
        btns = driver.find_elements(By.CSS_SELECTOR,
            "button[aria-expanded='false']"
        )
        for btn in btns[:30]:
            try:
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.25)
            except Exception:
                pass
    except Exception:
        pass


def extract_via_js(driver: webdriver.Chrome, js_selectors: List[str]) -> List[str]:
    """Try each CSS selector via JavaScript and return first non-empty result."""
    for sel in js_selectors:
        try:
            results = driver.execute_script(f"""
                var els = document.querySelectorAll('{sel}');
                return Array.from(els).map(e => e.innerText.trim()).filter(t => t.length > 0);
            """)
            if results:
                return results
        except Exception:
            continue
    return []


def scrape_course_page(driver: webdriver.Chrome, url: str) -> Dict[str, Any]:
    result = {"description": "", "syllabus": []}
    if not url:
        return result

    try:
        driver.get(url)
        try:
            WebDriverWait(driver, PAGE_LOAD_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "main, article"))
            )
        except TimeoutException:
            pass

        # Expand all collapsed accordions via JS
        driver.execute_script("""
            document.querySelectorAll('button[aria-expanded="false"]').forEach(btn => {
                try { btn.click(); } catch(e) {}
            });
        """)
        time.sleep(1.5)

        scroll_page(driver)

        # --- Description (JS) ---
        desc_selectors = [
            "[data-e2e='sdp-course-description']",
            "div.description__content",
            "div.about-section",
            "div.course-description",
        ]
        desc_results = extract_via_js(driver, desc_selectors)
        if desc_results:
            result["description"] = " ".join(desc_results)

        # Fallback — longest paragraph
        if not result["description"]:
            try:
                paras = driver.execute_script("""
                    return Array.from(document.querySelectorAll('p'))
                        .map(p => p.innerText.trim())
                        .filter(t => t.length > 100);
                """)
                if paras:
                    result["description"] = paras[0]
            except Exception:
                pass

        # --- Syllabus ---
        syllabus = []

        syllabus_selectors = [
            "[data-e2e='week-title']",
            "div.week-heading",
            "h3.module-name",
            "div[class*='WeekSingle'] h3",
            "div[class*='week'] h3",
            "div[class*='module'] h3",
            "div[class*='syllabus'] h3",
            "div[class*='syllabus'] h2",
        ]
        syllabus = extract_via_js(driver, syllabus_selectors)

        # Accordion fallback — take first line only (module title)
        if not syllabus:
            try:
                raw_items = driver.execute_script("""
                    return Array.from(document.querySelectorAll(
                        'button[aria-expanded], div[class*="accordion"], div[class*="Accordion"]'
                    )).map(e => e.innerText.trim()).filter(t => t.length > 3);
                """) or []
                noise_kws = [
                    "show less", "show more", "explore", "course details",
                    "hide info", "module details", "hours to complete",
                    "financial aid", "will i", "what will", "when will",
                    "is financial", "can i", "do i need", "how long",
                    "what background", "is this course",
                ]
                for item in raw_items:
                    first_line = item.split("\n")[0].strip()
                    if (
                        first_line
                        and 5 < len(first_line) < 150
                        and not any(kw in first_line.lower() for kw in noise_kws)
                    ):
                        syllabus.append(first_line)
            except Exception:
                pass

        # Specialization sub-course fallback
        if not syllabus:
            syllabus = extract_via_js(driver, [
                "h3.cds-CommonCard-title",
                "div.course-name",
                "h3[class*='title']",
            ])

        # Clean and dedupe
        noise_exact = {
            "explore", "show less", "show more", "course details",
            "hide info about module content", "module details",
            "community", "reviews", "overview",
        }
        noise_prefixes = (
            "there are", "will i", "what will", "when will", "is financial",
            "can i", "do i need", "how long", "what background", "is this course",
            "how often", "who should", "what background", "what comes",
            "is this training", "how does", "how is this", "what specific",
            "how will", "who is this", "what is the main", "what does a",
        )
        seen: set = set()
        syllabus_clean = []
        for s in (syllabus or []):
            if (
                s.lower() not in noise_exact
                and not any(s.lower().startswith(p) for p in noise_prefixes)
                and not s.replace("•", "").strip().endswith("hours")
                and not s.endswith("?")   # remove all FAQ questions
                and len(s) > 5
                and s not in seen
            ):
                seen.add(s)
                syllabus_clean.append(s)
        result["syllabus"] = syllabus_clean[:20]



    except WebDriverException as e:
        print(f"  [ERROR] Course page: {e}")

    return result


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
        flat.append({
            **{k: v for k, v in rec.items() if not isinstance(v, list)},
            "syllabus": " | ".join(rec.get("syllabus", [])),
        })
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
                    "course_url": "", "provider": "", "platform": "Coursera",
                    "description": "", "syllabus": [],
                })
                continue

            print(f"  Found : {course['title']}")
            print(f"  URL   : {course['url']}")

            time.sleep(random.uniform(3, 6))
            print("  Scraping course page...")
            page = scrape_course_page(driver, course["url"])

            print(f"  Description : {len(page['description'])} chars")
            print(f"  Syllabus    : {len(page['syllabus'])} modules — {page['syllabus'][:3]}")


            results.append({
                "topic_id":      gap["topic_id"],
                "topic_label":   gap["label"],
                "search_query":  gap["query"],
                "course_title":  course["title"],
                "course_url":    course["url"],
                "provider":      course.get("provider", ""),
                "platform":      "Coursera",
                "description":   page["description"],
                "syllabus":      page["syllabus"],

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

    empty_syl  = [r for r in results if not r["syllabus"]]
    if empty_syl:
        print(f"\n  [WARN] {len(empty_syl)} courses missing syllabus:")
        for r in empty_syl:
            print(f"    Topic {r['topic_id']}: {r['course_title']}")



if __name__ == "__main__":
    main()