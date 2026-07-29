"""
Workopolis Web Scraper
Scrapes job descriptions from workopolis.com
Saves all jobs into one JSON file

"""



import time, random, json, logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(logs_dir / "workopolis_scraper.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SEARCH_TERMS = {
    "data_science": ["Data Scientist"],
    "unrelated": [
        "Registered Nurse",
        "Primary School Teacher",
        "Accountant",
        "Human Resources Officer",
        "Sales Representative",
        "Restaurant Manager",
        "Chef",
        "Warehouse Supervisor",
        "Construction Supervisor",
        "Pharmacist",
    ],
}

# Cap collected jobs per category so "unrelated" stays comparable in size
# to the existing "data_science" group (~31 jobs previously collected from Workopolis).
CATEGORY_TARGETS = {
    "unrelated": 30,
}



BASE_URL = "https://www.workopolis.com/jobsearch/find-jobs?ak={}"

all_jobs = []


# ── DRIVER ─────────────────────────────────────────
def setup_driver():
    opts = Options()
    # opts.add_argument("--headless")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=opts)
    return driver


# ── GET URLS ───────────────────────────────────────
def get_job_urls(driver, term, pages=2):

    urls = []
    query = term.replace(" ", "+")

    for page in range(1, pages+1):

        url = f"{BASE_URL.format(query)}&pn={page}"
        driver.get(url)

        # allow redirects / JS load
        time.sleep(5)

        # try accept cookies if present
        try:
            btn = WebDriverWait(driver,3).until(
                EC.element_to_be_clickable((By.XPATH,"//button[contains(.,'Accept')]"))
            )
            btn.click()
            time.sleep(2)
        except:
            pass

        # WAIT FOR REAL JOB CARDS
        try:
            WebDriverWait(driver,12).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,"a[href*='/viewjob']"))
            )
        except:
            logging.warning(f"Blocked or empty page for {term} page {page}")
            continue

        soup = BeautifulSoup(driver.page_source,"html.parser")

        cards = soup.select("a[href*='/viewjob']")

        for c in cards:
            link = c.get("href")
            if link:
                if link.startswith("/"):
                    link = "https://www.workopolis.com"+link
                if link not in urls:
                    urls.append(link)

        logging.info(f"{term}: page {page} -> {len(cards)} cards")

        time.sleep(random.uniform(2,4))

    return urls


# ── SCRAPE JOB ─────────────────────────────────────
def scrape_job(driver,url,term):

    job={
        "source":"workopolis",
        "search_term":term,
        "url":url,
        "title":None,
        "company":None,
        "location":None,
        "description":None
    }

    try:
        driver.get(url)

        try:
            WebDriverWait(driver,10).until(
                EC.presence_of_element_located((By.TAG_NAME,"h1"))
            )
        except:
            logging.warning(f"Job page didn't load: {url}")

        soup=BeautifulSoup(driver.page_source,"html.parser")

        t=soup.find("h1")
        if t: job["title"]=t.get_text(strip=True)

        c=soup.select_one("[data-testid='company-name']")
        if c: job["company"]=c.get_text(strip=True)

        l=soup.select_one("[data-testid='job-location']")
        if l: job["location"]=l.get_text(strip=True)

        d=(soup.select_one("[data-testid='job-description']")
           or soup.find("div",{"id":"jobDescriptionText"})
           or soup.find("main"))

        if d:
            job["description"]=d.get_text(" ",strip=True)

        logging.info(f"Scraped: {job['title']}")

    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")

    return job


# ── SAVE JSON ──────────────────────────────────────
def save_json():
    out=Path("data/raw/workopolis_jobs.json")
    out.parent.mkdir(parents=True,exist_ok=True)

    with open(out,"w",encoding="utf-8") as f:
        json.dump(all_jobs,f,indent=2,ensure_ascii=False)

    print(f"\nSaved {len(all_jobs)} jobs → {out}")


# ── MAIN ───────────────────────────────────────────
def main():

    driver = setup_driver()
    print("Driver started")

    try:
        for category, titles in SEARCH_TERMS.items():

            print(f"\nCATEGORY: {category}")
            target = CATEGORY_TARGETS.get(category)
            # Cap applies per search term, not per category, so a single
            # early term (e.g. "Registered Nurse") can't exhaust the whole
            # category's quota before the other terms get a turn.
            per_term_target = target // len(titles) if target is not None else None
            category_jobs = []

            # ✅ THIS LOOP MUST BE INDENTED INSIDE CATEGORY LOOP
            for term in titles:

                print(f"Searching: {term}")

                urls = get_job_urls(driver, term, pages=2)

                print("Found", len(urls), "jobs")

                term_jobs = 0
                for u in urls:
                    if per_term_target is not None and term_jobs >= per_term_target:
                        print(f"Reached per-term target of {per_term_target} for '{term}', moving to next term")
                        break

                    job = scrape_job(driver, u, term)

                    if job["description"]:
                        job["category"] = category
                        category_jobs.append(job)
                        term_jobs += 1

                    time.sleep(random.uniform(1,3))

                # pause between search terms
                time.sleep(random.uniform(3,6))

            all_jobs.extend(category_jobs)

    finally:
        driver.quit()
        save_json()
        print("Done")


if __name__ == "__main__":
    
    main()