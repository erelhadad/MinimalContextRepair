import re
import json
import requests
import trafilatura
from pathlib import Path
from readability import Document
from bs4 import BeautifulSoup
from typing import Any, Dict, List
from urllib.parse import urlparse
import hashlib

from RagAdaptation.core.paths import DATA_DIR

def safe_filename_from_url(url: str, max_len: int = 180) -> str:
    u = urlparse(url)
    # ignore fragment (#...)
    base = f"{u.netloc}{u.path}"
    if not base:
        base = "download"

    # replace Windows-invalid filename chars + whitespace
    base = re.sub(r'[<>:"/\\|?*\s]+', "_", base).strip("._")

    # keep it short, but stable
    if len(base) > max_len:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        base = base[: max_len - 11] + "_" + h

    return base + ".txt"

def url_to_clean_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()
    html = r.text

    print("status:", r.status_code, "html_chars:", len(html))

    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
    if text and text.strip():
        print("trafilatura chars:", len(text))
        return text.strip()

    main_html = Document(html).summary()
    text2 = BeautifulSoup(main_html, "lxml").get_text(" ", strip=True)
    print("readability chars:", len(text2))
    return text2.strip()

def download_url(url: str) -> Path:
    text = url_to_clean_text(url)
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / safe_filename_from_url(url)
    out_path.write_text(text, encoding="utf-8")
    return out_path

if __name__ == "__main__":
    examples = [{"contradicting_url": "https://www.centerforfoodsafety.org/issues/311/ge-foods/ge-foods-are-not-proven-safe#"}]
    for example in examples:
        url = example["contradicting_url"]
        try:
            print("path saved:", download_url(url))
        except Exception as e:
            print("error:", e)