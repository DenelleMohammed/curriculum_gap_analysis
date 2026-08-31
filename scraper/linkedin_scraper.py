#!/usr/bin/env python3
"""
linkedin_scraper.py — Self-contained LinkedIn job scraper.
No external config.py or titles.py needed.

Scrapes Data Scientist job listings from LinkedIn and saves to:
  - data/raw/linkedin_jobs.csv
  - data/raw/linkedin_jobs.jsonl
"""

import json
import random
import re
import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    InvalidSessionIdException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# =========================
# Settings (edit here)
# =========================

OUTPUT_DIR       = Path("data/raw")
JOB_TITLES       = ["Data Scientist"]
US_LOCATIONS     = [
    "United States",
    "New York, NY",
    "San Francisco, CA",
    "Austin, TX",
    "Seattle, WA",
    "Remote",
]
RESULTS_PER_TITLE = 25
HEADLESS          = False   # Set False to see the browser / solve CAPTCHAs manually
MIN_SLEEP         = 4.0
MAX_SLEEP         = 9.0


# =========================
# Output paths
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH   = OUTPUT_DIR / "linkedin_jobs.csv"
JSONL_PATH = OUTPUT_DIR / "linkedin_jobs.jsonl"


# =========================
# Data model
# =========================

@dataclass
class JobRecord:
    source:           str
    region:           str
    title_query:      str
    location_query:   str
    collected_at_utc: str
    url:              str
    job_title:        Optional[str]
    company:          Optional[str]
    location:         Optional[str]
    description:      Optional[str]


# =========================
# Utilities
# =========================

def sleep_polite():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def build_driver() -> webdriver.Chrome:
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opts)


def clean_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def append_row_csv(path: Path, row: dict, fieldnames: list) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_row_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_seen_urls(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["url"])
        return set(df["url"].dropna().astype(str).tolist())
    except Exception:
        return set()


# =========================
# Parsing
# =========================

def parse_linkedin_job(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    title_el   = soup.select_one("h1")
    company_el = soup.select_one("a.topcard__org-name-link, span.topcard__flavor")
    loc_el     = soup.select_one("span.topcard__flavor--bullet")
    desc_el    = soup.select_one("div.show-more-less-html__markup")
    if not desc_el:
        desc_el = soup.select_one("div.description__text")

    return {
        "job_title":   clean_text(title_el.get_text(" ", strip=True)   if title_el   else None),
        "company":     clean_text(company_el.get_text(" ", strip=True)  if company_el else None),
        "location":    clean_text(loc_el.get_text(" ", strip=True)      if loc_el     else None),
        "description": clean_text(desc_el.get_text(" ", strip=True)     if desc_el    else None),
    }


# =========================
# URL collection
# =========================

def collect_job_urls(driver, query: str, location: str, limit: int) -> list:
    q = quote_plus(query)
    l = quote_plus(location)
    url = (
        f"https://www.linkedin.com/jobs/search/?keywords={q}"
        f"&location={l}&f_TPR=r2592000"
    )

    driver.get(url)
    sleep_polite()

    urls: set = set()

    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ul.jobs-search__results-list"))
        )
    except TimeoutException:
        print("  Timed out waiting for results list.")
        return []

    last_count  = 0
    scroll_tries = 0

    while len(urls) < limit and scroll_tries < 10:
        cards = driver.find_elements(By.CSS_SELECTOR, "ul.jobs-search__results-list li")

        for c in cards:
            try:
                a    = c.find_element(By.CSS_SELECTOR, "a.base-card__full-link")
                href = a.get_attribute("href")
                if href and "/jobs/view/" in href:
                    urls.add(href.split("?")[0])
            except NoSuchElementException:
                continue

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep_polite()

        if len(urls) == last_count:
            scroll_tries += 1
        else:
            scroll_tries = 0

        last_count = len(urls)

    return list(urls)[:limit]


# =========================
# Page collection
# =========================

def safe_get(driver, url: str) -> bool:
    try:
        driver.get(url)
        return True
    except (InvalidSessionIdException, WebDriverException):
        return False


def collect_job_pages(driver, urls: list, seen_urls: set) -> list:
    collected = []

    for u in urls:
        if u in seen_urls:
            continue

        print(f"    Visiting: {u}")

        if not safe_get(driver, u):
            print("    Driver session died.")
            break

        time.sleep(random.uniform(6, 12))

        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.5);")
            time.sleep(random.uniform(1, 3))
        except Exception:
            pass

        html   = driver.page_source
        parsed = parse_linkedin_job(html)
        collected.append((u, parsed))

    return collected


# =========================
# Main
# =========================

def main():
    seen_urls  = load_seen_urls(CSV_PATH)
    fieldnames = [f for f in JobRecord.__dataclass_fields__]
    print(f"Resuming. Already saved: {len(seen_urls)} jobs.\n")

    for title in JOB_TITLES:
        for loc in US_LOCATIONS:
            print(f"[LinkedIn] Query: '{title}' | Location: '{loc}'")

            driver = build_driver()

            try:
                urls = collect_job_urls(driver, title, loc, RESULTS_PER_TITLE)
                print(f"  Found {len(urls)} URLs")

                job_data = collect_job_pages(driver, urls, seen_urls)

                for u, parsed in job_data:
                    record = JobRecord(
                        source           = "linkedin",
                        region           = "US",
                        title_query      = title,
                        location_query   = loc,
                        collected_at_utc = datetime.now(timezone.utc).isoformat(),
                        url              = u,
                        job_title        = parsed.get("job_title"),
                        company          = parsed.get("company"),
                        location         = parsed.get("location"),
                        description      = parsed.get("description"),
                    )

                    row = asdict(record)
                    append_row_csv(CSV_PATH, row, fieldnames)
                    append_row_jsonl(JSONL_PATH, row)
                    seen_urls.add(u)
                    print(f"    Saved: {u}")

            except Exception as e:
                print(f"  Error during session: {e}")

            finally:
                driver.quit()

            cooldown = random.uniform(15, 35)
            print(f"  Cooling down {round(cooldown, 1)}s...\n")
            time.sleep(cooldown)

    print("Done.")


if __name__ == "__main__":
    main()