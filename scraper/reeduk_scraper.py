"""
Reed UK Job Scraper
Scrapes ICT job descriptions from reed.co.uk
Outputs ONE JSON file
Fully integrated logging (flushes in real-time)
"""

import json
import time
import random
import logging
from pathlib import Path
import shutil
import tempfile
import uuid
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup

# ── Logging Setup ───────────────────────────────────────────────
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logger = logging.getLogger("reed_scraper")
logger.setLevel(logging.INFO)

# File handler
file_log_path = logs_dir / "reed_scraper.log"
file_handler = logging.FileHandler(str(file_log_path), mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)


# ── Search Terms ───────────────────────────────────────────────
SEARCH_TERM = 'Data Scientist'
BASE_URL = "https://www.reed.co.uk/jobs/{}-jobs?pageno={}"

# ── Driver Setup ───────────────────────────────────────────────
def setup_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")

    # Unique temporary Chrome profile for each run
    temp_profile = os.path.join(tempfile.gettempdir(), f"chrome_tmp_profile_{uuid.uuid4().hex}")
    if os.path.exists(temp_profile):
        try:
            shutil.rmtree(temp_profile)
        except Exception as e:
            logger.warning(f"Could not remove old temp profile {temp_profile}: {e}")
    os.makedirs(temp_profile, exist_ok=True)
    opts.add_argument(f"--user-data-dir={temp_profile}")

    try:
        driver = webdriver.Chrome(options=opts)
        logger.info("ChromeDriver started successfully")
        return driver
    except Exception as e:
        logger.error(f"Failed to start ChromeDriver with temp profile {temp_profile}: {e}")
        raise

# ── Get Job URLs ───────────────────────────────────────────────
def get_job_urls(driver, term, pages=2):
    urls = []
    keyword = term.lower().replace(" ", "-")

    for page in range(1, pages + 1):
        url = BASE_URL.format(keyword, page)
        logger.info(f"Opening URL: {url}")
        driver.get(url)

        try:
            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "article")))
        except Exception:
            logger.warning(f"No listings loaded for {term} page {page}")
            continue

        soup = BeautifulSoup(driver.page_source, "html.parser")

        for a in soup.select("article a[href*='/jobs/']"):
            link = a.get("href")
            if link and "/jobs/" in link:
                if link.startswith("/"):
                    link = "https://www.reed.co.uk" + link
                if link not in urls:
                    urls.append(link)

        logger.info(f"Found {len(urls)} links so far for '{term}'")
        time.sleep(random.uniform(2, 4))

    return urls

# ── Scrape Job ───────────────────────────────────────────────
def scrape_job(driver, url, term):
    job = {
        "url": url,
        "searched_role": term,
        "title": None,
        "company": None,
        "location": None,
        "description": None,
        "source": "reed"
    }

    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception:
        logger.warning(f"Page did not load: {url}")
        return job

    soup = BeautifulSoup(driver.page_source, "html.parser")

    title_tag = soup.find("h1")
    if title_tag:
        job["title"] = title_tag.get_text(strip=True)

    company_tag = soup.select_one("[data-qa='company-name'], .company")
    if company_tag:
        job["company"] = company_tag.get_text(strip=True)

    location_tag = soup.select_one("[data-qa='job-location'], .location")
    if location_tag:
        job["location"] = location_tag.get_text(strip=True)

    desc_tag = (
        soup.select_one("#jobDescription")
        or soup.select_one(".job-description")
        or soup.select_one("[data-qa='job-description']")
        or soup.find("main")
    )

    if desc_tag:
        job["description"] = desc_tag.get_text(" ", strip=True)
    else:
        logger.warning(f"No description for {url}")

    logger.info(f"Collected job: {job['title']} at {job.get('company', 'Unknown')}")
    return job

# ── Save JSON ───────────────────────────────────────────────
def save_json(jobs):
    out = Path("data/raw/reed_jobs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(jobs)} jobs to {out}")

# ── Main ───────────────────────────────────────────────
def main():
    driver = setup_driver()
    all_jobs = []

    try:
        term = SEARCH_TERM
        logger.info(f"Searching for term: {term}")
        urls = get_job_urls(driver, term, pages=2)
        logger.info(f"Total URLs found for '{term}': {len(urls)}")

        for i, link in enumerate(urls, start=1):
            logger.info(f"Scraping job {i}/{len(urls)}")
            try:
                job = scrape_job(driver, link, term)
                if job["description"]:
                    job["category"] = "data_science"
                    all_jobs.append(job)
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                logger.error(f"Error scraping job {link}: {e}")

    finally:
        driver.quit()
        save_json(all_jobs)
        logger.info("Scraping complete. Driver closed.")
        logging.shutdown()  # Flush all logs immediately

if __name__ == "__main__":
    main()