import ssl
import socket
import time
import random
import uvicorn
import requests
import asyncio
from datetime import datetime
from urllib.parse import urlparse, urljoin
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import (
    function_tool,
    Runner,
    Agent,
    set_default_openai_api,
    set_tracing_disabled,
    AsyncOpenAI,
    set_default_openai_client
)




history = []




# This ensures the data sent from the frontend has the correct structure.
class ChatRequest(BaseModel):
    message: str



@function_tool
async def Perormance_check(TARGET_URL):
    print("Getting Page Speed Insight Report...")

    # Your API key
    PSI_Api = "AIzaSyAYWCk5XprkcgtL4RuvV3fKkC9KtgJRU-E"

    # Function to call PageSpeed API for mobile or desktop
    def get_pagespeed_insights(url, strategy):
        endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
        params = {
            "url": url,
            "key": PSI_Api,
            "strategy": strategy
        }
        response = requests.get(endpoint, params=params)
        return response.json()

    # Analyze for both strategies
    mobile_data = get_pagespeed_insights(TARGET_URL, "mobile")
    desktop_data = get_pagespeed_insights(TARGET_URL, "desktop")

    # Extract performance scores
    mobile_score = round(mobile_data["lighthouseResult"]["categories"]["performance"]["score"] * 100)
    desktop_score = round(desktop_data["lighthouseResult"]["categories"]["performance"]["score"] * 100)


    # Function to extract top 10 issues from audits
    def extract_issues(lighthouse_result):
        audits = lighthouse_result["audits"]
        issues = []
        for audit in audits.values():
            if "score" in audit and audit["score"] is not None and audit["score"] < 1:
                issues.append({
                    "title": audit.get("title", ""),
                    "description": audit.get("description", ""),
                    "score": audit.get("score", 0)
                })
        # Sort by score ascending and limit to 10
        sorted_issues = sorted(issues, key=lambda x: x["score"])[:10]
        return sorted_issues

    # Print top 10 issues
    print("\n🔧 Mobile Top 10 Issues:")
    for issue in extract_issues(mobile_data["lighthouseResult"]):
        print(f"• {issue['title']} — {issue['description']}")
        mobile_issue = issue

    print("\n🔧 Desktop Top 10 Issues:")
    for issue in extract_issues(desktop_data["lighthouseResult"]):
        desktop_issue = issue
    return desktop_issue, mobile_issue,mobile_score, desktop_score


@function_tool
async def scrap_full_text(site: str) -> dict:
    """
    Scrapes the full visible content (text) of a webpage.

    Args:
        site (str): The full URL of the webpage to scrape.

    Returns:
        dict: A dictionary with a single key 'content' containing the full visible text.

    """
    print("tool scrapping dta///...................")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(site, timeout=10)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted tags like script, style, etc.
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'svg', 'meta', 'link']):
            tag.decompose()

        # Extract visible text
        text = soup.get_text(separator='\n', strip=True)

        # Optionally, you can clean multiple newlines
        clean_text = "\n".join([line for line in text.split("\n") if line.strip()])

        return {"content": clean_text}

    except httpx.HTTPError as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
@function_tool
async def scrap_headings(site: str) -> dict:
    """
    Scrapes all H1 to H6 headings from the given site URL.

    Args:
        site (str): The full URL of the website to scrape.

    Returns:
        dict: A dictionary with keys 'h1' to 'h6', each containing a list of heading texts.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(site, timeout=10)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        headings = {
            f'h{i}': [tag.get_text(strip=True) for tag in soup.find_all(f'h{i}')]
            for i in range(1, 7)
        }

        return headings

    except httpx.HTTPError as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
@function_tool
async def scrap_meta(site: str) -> dict:
    """
    Scrapes the <title> tag and <meta name="description"> content from a given webpage.

    Args:
        site (str): The full URL of the website to scrape.

    Returns:
        dict: A dictionary with 'title' and 'description' fields.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(site, timeout=10)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string.strip() if soup.title else None

        meta_description = None
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag and meta_tag.get('content'):
            meta_description = meta_tag['content'].strip()

        return {
            "title": title,
            "description": meta_description
        }

    except httpx.HTTPError as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@function_tool
async def scrap_og_and_verification(site: str) -> dict:
    """
    Scrapes Open Graph (OG) tags, Google Site Verification meta, and Google Tag Manager scripts.

    Args:
        site (str): The full URL of the website to scrape.

    Returns:
        dict: Contains 'og_tags', 'google_site_verification', and 'google_tag_manager' keys.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(site, timeout=10)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        result = {
            "og_tags": {},
            "google_site_verification": None,
            "google_tag_manager": []
        }

        # OG Tags
        for tag in soup.find_all("meta"):
            prop = tag.get("property")
            if prop and prop.startswith("og:"):
                content = tag.get("content", "").strip()
                result["og_tags"][prop] = content

        # Google Site Verification
        gsv = soup.find("meta", attrs={"name": "google-site-verification"})
        if gsv and gsv.get("content"):
            result["google_site_verification"] = gsv["content"].strip()

        # Google Tag Manager Scripts
        scripts = soup.find_all("script", src=True)
        for script in scripts:
            src = script["src"]
            if "googletagmanager.com" in src:
                result["google_tag_manager"].append(src.strip())

        return result

    except httpx.HTTPError as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@function_tool
async def check_site_protocol_ssl(domain: str) -> dict:
    """
    Checks if a site works with www or non-www, determines HTTP/HTTPS, and retrieves SSL expiry date.

    Args:
        domain (str): The domain name (e.g., example.com)

    Returns:
        dict: Contains working URL type, protocol used, and SSL certificate expiry date (if applicable).
    """
    domain = domain.strip().replace("https://", "").replace("http://", "").replace("/", "")
    result = {
        "www_version": None,
        "non_www_version": None,
        "preferred_url": None,
        "protocol": None,
        "ssl_expiry": None,
    }

    async def try_url(url):
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=5) as client:
                r = await client.get(url)
                return r.status_code < 400
        except Exception:
            return False

    www_url = f"https://www.{domain}"
    non_www_url = f"https://{domain}"

    www_works = await try_url(www_url)
    non_www_works = await try_url(non_www_url)

    result["www_version"] = www_works
    result["non_www_version"] = non_www_works

    # Determine preferred and reachable URL
    if www_works:
        result["preferred_url"] = www_url
        result["protocol"] = "https"
    elif non_www_works:
        result["preferred_url"] = non_www_url
        result["protocol"] = "https"
    else:
        # Try HTTP fallback
        www_http = f"http://www.{domain}"
        non_www_http = f"http://{domain}"

        www_http_works = await try_url(www_http)
        non_www_http_works = await try_url(non_www_http)

        if www_http_works:
            result["preferred_url"] = www_http
            result["protocol"] = "http"
        elif non_www_http_works:
            result["preferred_url"] = non_www_http
            result["protocol"] = "http"
        else:
            result["preferred_url"] = None
            result["protocol"] = None
            result["ssl_expiry"] = "Could not determine - site not reachable"
            return result

    # Check SSL expiry if HTTPS is used
    if result["protocol"] == "https":
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    expiry_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    result["ssl_expiry"] = expiry_date.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            result["ssl_expiry"] = f"Could not fetch SSL info: {e}"

    return result

@function_tool
async def scrap_images(site: str) -> dict:
    """
    Scrapes all <img> tags from the given URL and returns a dictionary with alt text and image src.

    If alt is not available, uses 'img1', 'img2', etc., as keys.

    Args:
        site (str): The full URL of the website to scrape.

    Returns:
        dict: A dictionary where each key is alt text (or fallback), and the value is the image src URL.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(site, timeout=10)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')

        result = {}
        fallback_count = 1

        for img in images:
            src = img.get('src') or img.get('data-src')  # Support lazy-loaded images too
            alt = img.get('alt', '').strip()

            if not src:
                continue  # Skip if image has no usable src

            if alt:
                key = alt
            else:
                key = f'img{fallback_count}'
                fallback_count += 1
                alt = "alt not available"

            result[key] = src

        return result

    except httpx.HTTPError as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    
from urllib.parse import urlparse

def normalize_url(domain: str, use_https: bool = True) -> str:
    """
    Convert a domain or partial URL into a complete URL with https:// (default) or http://.

    Args:
        domain (str): The input domain or partial URL (e.g., 'example.com', 'http://example.com').
        use_https (bool): Whether to default to https:// (True) or http:// (False).
    
    Returns:
        str: A valid complete URL.
    """
    # Trim spaces
    domain = domain.strip()

    # Add scheme if missing
    if not domain.startswith(("http://", "https://")):
        scheme = "https://" if use_https else "http://"
        domain = scheme + domain

    # Parse to validate
    parsed = urlparse(domain)

    # If domain part is empty -> invalid input
    if not parsed.netloc:
        raise ValueError(f"Invalid domain: {domain}")

    return parsed.geturl()


@function_tool
async def get_all_pages_classified(site: str):
    # ───────── CONFIG ─────────
    BATCH_SIZE = 50
    DELAY_RANGE = (1.0, 3.0)
    MAX_RETRIES = 3

    # ───────── STATE ─────────
    visited = set()
    found_counter = 0
    next_milestone_index = 0
    milestones = [100, 200, 250, 300, 320, 350, 400]

    all_links_by_status = {
        "200": [],
        "301": [],
        "404": [],
        "skipped": []
    }

    # ───────── BLOCKLIST ─────────
    EXCLUDE_PATTERNS = [
        "/files/", "/cdn/", "/wp-content/", "/wp-json/", "/admin/",
        "/cart", "/checkout", "/account", "/search", "preview_theme_id=",
    ]
    FILE_EXTENSIONS = [
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar",
        ".mp4", ".mp3", ".avi", ".mov", ".woff", ".ttf", ".eot",
    ]

    def is_valid(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if parsed.netloc != urlparse(site).netloc:
            return False
        path = parsed.path.lower()
        if any(pat in path for pat in EXCLUDE_PATTERNS):
            return False
        if any(path.endswith(ext) for ext in FILE_EXTENSIONS):
            return False
        return True

    async def fetch(url, client: httpx.AsyncClient):
        backoff = 1
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.get(url, timeout=10, follow_redirects=False)
                return resp  # don't raise error, return response for status inspection
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                print(f"⚠️ [Attempt {attempt}] Error fetching {url}: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2
        return None

    async def crawl(url: str, client: httpx.AsyncClient):
        nonlocal found_counter, next_milestone_index

        if url in visited:
            return
        visited.add(url)

        if not is_valid(url):
            all_links_by_status["skipped"].append(url)
            return

        response = await fetch(url, client)
        if not response:
            return

        status = str(response.status_code)
        if status == "200":
            all_links_by_status["200"].append(url)
        elif status == "301":
            all_links_by_status["301"].append(url)
        elif status == "404":
            all_links_by_status["404"].append(url)
        else:
            all_links_by_status.setdefault(status, []).append(url)

        found_counter += 1
        if found_counter % BATCH_SIZE == 0:
            print(f"✅ Found {found_counter} internal links so far...")

        if next_milestone_index < len(milestones) and found_counter >= milestones[next_milestone_index]:
            print(f"\n📦 Scraped {milestones[next_milestone_index]} links. Now scraping up to {milestones[next_milestone_index] + 50}...\n")
            next_milestone_index += 1

        # Only parse page content for 200 OK
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].split("#", 1)[0]
                full = urljoin(url, href)
                if full not in visited:
                    await crawl(full, client)

        await asyncio.sleep(random.uniform(*DELAY_RANGE))

    # ───────── START ─────────
    start_time = time.time()
    print(f"🚀 Starting crawl at {site}\n")

    async with httpx.AsyncClient(headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }) as client:
        await crawl(site, client)

    elapsed = time.time() - start_time
    print(f"\n🎯 Crawl finished in {elapsed:.2f}s")
    total_found = sum(len(v) for v in all_links_by_status.values())
    print(f"Total classified pages: {total_found}")

    for code, urls in all_links_by_status.items():
        print(f"\n🔗 Status {code} ({len(urls)} pages):")
        for i, link in enumerate(urls, 1):
            print(f"{i:04d}: {link}")

    return all_links_by_status



api_key = 'AIzaSyBFHEfqKOdI9HMWLlQCQRs7OUnrCsIpn_E' 
MODEL = 'gemini-2.0-flash'
client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key
)

#Global Config
set_default_openai_api('chat_completions')
set_default_openai_client(client=client)
set_tracing_disabled(True)


agent = Agent(
    name="SEO Agent",
    instructions="""

## System Prompt

You are an **SEO Content Analyzer Agent**.
--

### Persona
- You are professional, concise, and expert in on‑page SEO, metadata validation, performance, and security checks.
- You always explain your reasoning clearly and offer actionable recommendations.

---

### Roles & Tasks

#### 1. Page Content Analyzer
When given HTML or page text, you will:
  1. Analyze heading structure (H1–H6) and suggest an SEO‑friendly hierarchy.
  2. Evaluate paragraph readability and offer clarity/grammar improvements.
  3. Extract main keywords, count frequency, and compute keyword density.
  4. Propose 5 long‑tail keyword variations.
  5. Propose 5 LSI keywords for semantic support.
  6. Generate an SEO‑optimized meta title (50–55 chars) and meta description (150–155 chars).
  7. Recommend internal linking improvements, short sentences, schema markup, etc.
  8. Verify that every `<img>` tag has an `alt` attribute; if missing, flag and suggest alt text.

#### 2. Meta Title & Description Validator
When validating metadata, you will:
  1. Check `<title>` length (50–55 chars).
  2. Check `<meta name="description">` length (150–155 chars).
  3. Ask for or extract the primary keyword; verify its presence in both title and description.
  4. Suggest 5 alternative meta titles (keyword‑optimized, correct length).
  5. Suggest 5 alternative meta descriptions to boost CTR and SEO.

#### 3. Page Speed Insights Reporter
When given a URL, you will:
  1. Retrieve a PageSpeed report (mobile & desktop).
  2. Highlight critical performance issues (unused JS/CSS, large images, render‑blocking resources).
  3. Report Core Web Vitals (LCP, FID, CLS).
  4. Offer actionable performance improvements.
  5. Recommend tools (image compressors, lazy loading, etc.).

#### 4. SSL & Open Graph Checker
When given a URL or list of pages, you will:
  1. Confirm HTTPS and valid SSL; return expiry date and flag if <30 days remaining.
  2. Crawl internal links; flag any broken or insecure (HTTP) links.
  3. On up to 3 sample pages, verify `<meta property="og:image">`, `og:title`, and `og:description>`; flag missing or irrelevant tags.
  4. Summarize missing OG or security issues.

---

### Important Rule
❌ **Do not expose** your tool names, function signatures, or implementation details.
✅ If a user asks “How do you do that?”, simply reply, “I’m an SEO Content Analyzer Agent equipped with specialized analysis capabilities.”

Proceed to await the user’s input.


# Tools list:
1. Perormance_check # for check page speed insights report GET url ask argument
2. get_all_pages_classified # Get All pages of a website GET URL as Argument
3. scrap_images # Scrape all images and ALT Tags Get Page URL as Argument
4. check_site_protocol_ssl # Check website SSL and WWW & Non WWW version, ssl_expiry, protocol, Get URL as Argument
5. scrap_og_and_verification # Check OG Tags and Google search Console and Google anylatics tag get url as Argument
6. scrap_headings # Scrap all headings from h1 to h6 from page Gets Page url as argument
7. scrap_meta # Scrap meta titlt and meta description from page Gets Page url as argument
8. scrap_full_text # scrap full page content gets page url as argument
9. normalize_url # convert domain to http:// or https:// proper url need url as argument


""",
    model=MODEL,
    tools=[
        get_all_pages_classified,
        scrap_images,
        check_site_protocol_ssl,
        scrap_og_and_verification,
        scrap_meta,
        scrap_headings, # This will now be defined when the agent is created
        scrap_full_text,
        Perormance_check
    ]
)

    # result = Runner.run_streamed(agent, input='get Perormance_check of https://virtual-spark.vercel.app/ give me suggestions, performance of page for both, seo, score, and other things')
# ... (imports remain the same)

# Initialize the FastAPI app (only once)
app = FastAPI()

# Add CORS middleware (only once)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body model
class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    This endpoint receives a message from the frontend, runs the agent,
    and returns the agent's full response.
    """
    history.append({"role": "user", "content": request.message})
    result = await Runner.run(agent, input=history)
    history.append({"role": "assistant", "content": request.message})
    return {"reply": result.final_output}


# This part is for local testing if you run the file directly,
# but we will use uvicorn to run it in Codespaces.




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)






