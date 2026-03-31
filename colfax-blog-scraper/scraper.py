#!/usr/bin/env python3
"""
Scraper for https://research.colfax-intl.com/ blog.
Converts each blog post to a markdown file with images downloaded locally.
Preserves: images, tables (as HTML), formulas (LaTeX), code blocks, headings, lists.
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
import urllib.parse
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag, Comment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://research.colfax-intl.com"
OUTPUT_DIR = Path(__file__).parent / "blogs"
MAX_PAGES = 10  # pagination upper bound (will stop early if no more pages)
REQUEST_DELAY = 1.0  # seconds between requests
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch(url: str) -> Optional[requests.Response]:
    """Fetch a URL with retries."""
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(2 ** attempt)
    log.error("Failed to fetch %s after 3 attempts", url)
    return None


def slugify(text: str) -> str:
    """Create filesystem-safe slug from text."""
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text[:120]


def download_image(url: str, dest_dir: Path) -> Optional[str]:
    """Download an image and return the local filename."""
    if not url:
        return None
    # Normalize URL
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = BASE_URL + url

    # Try to get full-size image by stripping resize params from wp.com CDN
    parsed = urllib.parse.urlparse(url)
    if "wp.com" in parsed.netloc:
        # Remove resize/quality params to get original
        clean_url = urllib.parse.urlunparse(parsed._replace(query=""))
        url = clean_url

    # Generate filename from URL
    parsed = urllib.parse.urlparse(url)
    path_part = parsed.path.rstrip("/")
    ext = os.path.splitext(path_part)[1] or ".png"
    # Use hash to avoid collisions
    name_base = os.path.basename(path_part)
    if not name_base or name_base == "/":
        name_base = hashlib.md5(url.encode()).hexdigest()[:12]
    # Clean filename
    name_base = re.sub(r'[^\w.-]', '_', name_base)
    filename = name_base if name_base.endswith(ext) else name_base + ext

    dest = dest_dir / filename
    if dest.exists():
        return filename

    resp = fetch(url)
    if resp is None:
        return None
    dest.write_bytes(resp.content)
    log.info("  Downloaded image: %s", filename)
    return filename


# ---------------------------------------------------------------------------
# HTML to Markdown converter (custom, preserves tables/math/code)
# ---------------------------------------------------------------------------

class HTML2Markdown:
    """Convert a BeautifulSoup element tree to Markdown, preserving:
    - Headings (h1-h6)
    - Paragraphs
    - Lists (ul, ol, nested)
    - Code blocks (pre/code)
    - Images (download locally)
    - Tables (keep as raw HTML)
    - LaTeX formulas (MathJax)
    - Blockquotes
    - Links
    - Bold, italic, inline code
    """

    def __init__(self, img_dir: Path):
        self.img_dir = img_dir
        self._list_depth = 0
        self._ol_counters: list[int] = []

    def convert(self, element: Tag) -> str:
        md = self._process_element(element)
        # Truncate at STOP marker (social sharing section)
        stop_idx = md.find("<!-- STOP -->")
        if stop_idx != -1:
            md = md[:stop_idx]
        # Clean up excessive blank lines
        md = re.sub(r'\n{3,}', '\n\n', md)
        return md.strip() + "\n"

    def _process_children(self, element: Tag) -> str:
        parts = []
        for child in element.children:
            result = self._process_element(child)
            if "<!-- STOP -->" in result:
                parts.append(result[:result.find("<!-- STOP -->")])
                parts.append("<!-- STOP -->")
                break
            parts.append(result)
        return "".join(parts)

    def _process_element(self, el) -> str:
        if isinstance(el, Comment):
            return ""
        if isinstance(el, NavigableString):
            text = str(el)
            # Don't strip text inside pre tags
            parent = el.parent
            if parent and parent.name in ("pre", "code"):
                return text
            # Collapse whitespace for normal text
            text = re.sub(r'[ \t]+', ' ', text)
            return text

        if not isinstance(el, Tag):
            return ""

        tag = el.name

        # --- Skip non-content elements ---
        if tag in ("script", "style", "nav", "header", "footer", "aside",
                    "noscript", "iframe"):
            return ""

        # --- Skip social sharing / like widgets ---
        el_classes = el.get("class", []) if hasattr(el, 'get') else []
        el_class_str = " ".join(el_classes) if el_classes else ""
        if any(skip in el_class_str for skip in [
            "sd-sharing", "sd-block", "sd-social", "sharedaddy",
            "post-likes-widget", "likes-widget", "jp-relatedposts",
            "comment-respond", "wp-block-post-comments",
            "robots-nocontent", "wp-block-post-date",
            "taxonomy-category", "wp-block-post-terms",
        ]):
            return ""
        # Skip "Share this:" and "Like this:" heading sections
        if tag in ("h3", "h4", "h5") and hasattr(el, 'get_text'):
            heading_text = el.get_text(strip=True).lower()
            if heading_text in ("share this:", "like this:", "related"):
                # Also skip everything after this heading until next major section
                return "<!-- STOP -->"

        # --- Headings ---
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            text = self._get_text(el).strip()
            if not text:
                return ""
            return f"\n\n{'#' * level} {text}\n\n"

        # --- Paragraphs ---
        if tag == "p":
            content = self._inline_content(el)
            if not content.strip():
                return ""
            return f"\n\n{content}\n\n"

        # --- KaTeX display block (div wrapping <pre> with LaTeX) ---
        if tag == "div" and "katex-eq" in el_class_str:
            latex = el.get_text().strip()
            if el.get("data-katex-display") == "true":
                return f"\n\n$$\n{latex}\n$$\n\n"
            return f"${latex}$"

        # --- Code blocks (but NOT if parent is katex-eq) ---
        if tag == "pre":
            parent = el.parent
            parent_classes = " ".join(parent.get("class", [])) if parent and hasattr(parent, 'get') else ""
            if "katex-eq" in parent_classes:
                # This is LaTeX inside a KaTeX block, not a code block
                return el.get_text()
            return self._convert_pre(el)

        # Skip bare <code> that's already inside pre (handled by _convert_pre)
        if tag == "code":
            parent = el.parent
            if parent and parent.name == "pre":
                return self._process_children(el)
            # Inline code
            text = el.get_text()
            if not text:
                return ""
            return f"`{text}`"

        # --- Tables: keep as HTML ---
        if tag == "table":
            return self._convert_table(el)

        # Wrapper figures for tables (wp-block-table)
        if tag == "figure" and el.get("class") and "wp-block-table" in el.get("class", []):
            table = el.find("table")
            if table:
                result = self._convert_table(table)
                caption = el.find("figcaption")
                if caption:
                    cap_text = self._get_text(caption).strip()
                    if cap_text:
                        result += f"\n<p align=\"center\"><em>{cap_text}</em></p>\n"
                return result
        # Div directly wrapping a table (only if table is a direct child)
        if tag == "div" and el.get("class") and "wp-block-table" in el.get("class", []):
            table = el.find("table", recursive=False)
            if table:
                result = self._convert_table(table)
                return result

        # --- Images ---
        if tag == "img":
            return self._convert_img(el)

        # --- Figure ---
        if tag == "figure":
            return self._convert_figure(el)

        # --- Lists ---
        if tag == "ul":
            return self._convert_list(el, ordered=False)
        if tag == "ol":
            return self._convert_list(el, ordered=True)
        if tag == "li":
            return self._process_children(el)

        # --- Blockquote ---
        if tag == "blockquote":
            content = self._process_children(el).strip()
            lines = content.split("\n")
            quoted = "\n".join(f"> {line}" for line in lines)
            return f"\n\n{quoted}\n\n"

        # --- Links ---
        if tag == "a":
            href = el.get("href", "")
            text = self._inline_content(el).strip()
            if not text:
                text = href
            if not href or href.startswith("#"):
                return text
            return f"[{text}]({href})"

        # --- Inline formatting ---
        if tag in ("strong", "b"):
            text = self._inline_content(el)
            if not text.strip():
                return text
            return f"**{text.strip()}**"

        if tag in ("em", "i"):
            text = self._inline_content(el)
            if not text.strip():
                return text
            return f"*{text.strip()}*"

        if tag == "br":
            return "  \n"

        if tag == "hr":
            return "\n\n---\n\n"

        # --- Definition lists ---
        if tag == "dl":
            return self._process_children(el)
        if tag == "dt":
            text = self._inline_content(el).strip()
            return f"\n\n**{text}**\n\n"
        if tag == "dd":
            text = self._inline_content(el).strip()
            return f": {text}\n\n"

        # --- MathJax / KaTeX / LaTeX ---
        # KaTeX equations (span.katex-eq with data-katex-display attribute)
        if tag == "span" and el.get("class"):
            classes = el.get("class", [])
            if "katex-eq" in classes:
                latex = el.get_text().strip()
                if el.get("data-katex-display") == "true":
                    return f"\n\n$$\n{latex}\n$$\n\n"
                else:
                    return f"${latex}$"
            # MathJax rendered elements
            if "MathJax" in classes or "mathjax" in " ".join(classes).lower():
                script = el.find("script", {"type": "math/tex"})
                if script:
                    latex = script.get_text()
                    return f"${latex}$"
            if "katex" in " ".join(classes).lower():
                annotation = el.find("annotation", {"encoding": "application/x-tex"})
                if annotation:
                    latex = annotation.get_text()
                    return f"${latex}$"

        # MathJax script tags
        if tag == "script" and el.get("type") in ("math/tex", "math/tex; mode=display"):
            latex = el.get_text()
            if el.get("type") == "math/tex; mode=display":
                return f"\n\n$$\n{latex}\n$$\n\n"
            return f"${latex}$"

        # --- Div handling for MathJax display math ---
        if tag == "div":
            classes = el.get("class", [])
            class_str = " ".join(classes) if classes else ""
            # MathJax display containers
            if "MathJax_Display" in class_str or "mathjax" in class_str.lower():
                script = el.find("script", {"type": re.compile(r"math/tex")})
                if script:
                    latex = script.get_text()
                    return f"\n\n$$\n{latex}\n$$\n\n"

        # --- wp-block-code ---
        if tag == "div" and el.get("class") and "wp-block-code" in el.get("class", []):
            pre = el.find("pre")
            if pre:
                return self._convert_pre(pre)

        # --- Default: recurse into children ---
        return self._process_children(el)

    def _get_text(self, el: Tag) -> str:
        """Get clean text content."""
        return el.get_text(separator=" ", strip=True)

    def _inline_content(self, el: Tag) -> str:
        """Process inline content of an element."""
        parts = []
        for child in el.children:
            parts.append(self._process_element(child))
        return "".join(parts)

    def _convert_pre(self, el: Tag) -> str:
        """Convert pre/code block to markdown fenced code block."""
        code_el = el.find("code")
        if code_el:
            code_text = code_el.get_text()
            # Try to detect language from class
            lang = ""
            classes = code_el.get("class", [])
            for cls in classes:
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
                    break
                elif cls.startswith("lang-"):
                    lang = cls.replace("lang-", "")
                    break
            if not lang:
                # Check parent pre for classes
                pre_classes = el.get("class", [])
                for cls in pre_classes:
                    if cls.startswith("language-"):
                        lang = cls.replace("language-", "")
                        break
        else:
            code_text = el.get_text()
            lang = ""

        # Remove trailing whitespace per line but preserve structure
        lines = code_text.split("\n")
        # Remove leading/trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        code_text = "\n".join(lines)

        return f"\n\n```{lang}\n{code_text}\n```\n\n"

    def _convert_table(self, table: Tag) -> str:
        """Keep tables as raw HTML embedded in markdown."""
        # Clean up the table HTML
        # Remove unnecessary attributes but keep structure
        table_html = str(table)
        # Remove excessive whitespace in HTML
        table_html = re.sub(r'\n\s*\n', '\n', table_html)
        return f"\n\n{table_html}\n\n"

    def _convert_img(self, el: Tag) -> str:
        """Convert image: download and create local reference."""
        src = el.get("data-src") or el.get("src") or ""
        alt = el.get("alt", "")

        if not src:
            return ""

        # WordPress LaTeX images (latex.php) — convert to LaTeX notation
        if "latex.php" in src or "s0.wp.com/latex" in src:
            latex = alt.strip() if alt else ""
            if latex:
                # Check if parent suggests display mode
                parent = el.parent
                if parent and parent.name == "p":
                    # Inline LaTeX
                    return f"${latex}$"
                return f"${latex}$"
            return ""

        # Skip tracking pixels, tiny images
        width = el.get("width")
        height = el.get("height")
        if width and height:
            try:
                if int(width) <= 1 or int(height) <= 1:
                    return ""
            except ValueError:
                pass

        # Download image
        local_name = download_image(src, self.img_dir)
        if local_name:
            return f"\n\n![{alt}](images/{local_name})\n\n"
        else:
            # Fallback to original URL
            return f"\n\n![{alt}]({src})\n\n"

    def _convert_figure(self, el: Tag) -> str:
        """Convert figure element (image + optional caption)."""
        parts = []
        img = el.find("img")
        if img:
            parts.append(self._convert_img(img))

        caption = el.find("figcaption")
        if caption:
            cap_text = self._inline_content(caption).strip()
            if cap_text:
                parts.append(f"\n<p align=\"center\"><em>{cap_text}</em></p>\n")

        return "".join(parts) if parts else self._process_children(el)

    def _convert_list(self, el: Tag, ordered: bool) -> str:
        """Convert list with proper nesting."""
        self._list_depth += 1
        if ordered:
            self._ol_counters.append(0)

        indent = "  " * (self._list_depth - 1)
        items = []

        for child in el.children:
            if not isinstance(child, Tag):
                continue
            if child.name != "li":
                continue

            if ordered:
                self._ol_counters[-1] += 1
                marker = f"{self._ol_counters[-1]}."
            else:
                marker = "-"

            # Process li content: separate nested lists from inline content
            inline_parts = []
            nested_parts = []
            for li_child in child.children:
                if isinstance(li_child, Tag) and li_child.name in ("ul", "ol"):
                    nested_parts.append(self._process_element(li_child))
                else:
                    inline_parts.append(self._process_element(li_child))

            inline_text = "".join(inline_parts).strip()
            # Remove newlines within the inline part of list item
            inline_text = re.sub(r'\n{2,}', ' ', inline_text)
            inline_text = inline_text.strip()

            item_str = f"{indent}{marker} {inline_text}"
            for nested in nested_parts:
                item_str += "\n" + nested.rstrip("\n")

            items.append(item_str)

        self._list_depth -= 1
        if ordered:
            self._ol_counters.pop()

        result = "\n".join(items)
        if self._list_depth == 0:
            result = "\n\n" + result + "\n\n"
        return result


# ---------------------------------------------------------------------------
# Blog post extraction
# ---------------------------------------------------------------------------

def extract_post_content(soup: BeautifulSoup, url: str, img_dir: Path) -> tuple[str, str, str]:
    """Extract blog post title, date, and markdown content.
    Returns (title, date, markdown_content).
    """
    # Title
    title_el = soup.find("h1", class_=re.compile(r"wp-block-post-title|entry-title"))
    if not title_el:
        title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else "Untitled"

    # Date
    date_el = soup.find("time", class_=re.compile(r"wp-block-post-date|entry-date"))
    if not date_el:
        date_el = soup.find("time")
    date_str = date_el.get_text(strip=True) if date_el else ""

    # Main content
    content_el = soup.find("div", class_=re.compile(r"wp-block-post-content|entry-content"))
    if not content_el:
        # Try article tag
        content_el = soup.find("article")
    if not content_el:
        content_el = soup.find("main")

    if not content_el:
        log.warning("Could not find main content for: %s", url)
        return title, date_str, ""

    converter = HTML2Markdown(img_dir=img_dir)
    md_content = converter.convert(content_el)

    return title, date_str, md_content


# ---------------------------------------------------------------------------
# Discover all blog post URLs from the index pages
# ---------------------------------------------------------------------------

def get_all_post_urls() -> list[dict]:
    """Crawl paginated index to collect all blog post URLs and titles."""
    all_posts = []
    seen_urls = set()

    for page_num in range(1, MAX_PAGES + 1):
        if page_num == 1:
            url = BASE_URL + "/"
        else:
            url = f"{BASE_URL}/?query-33-page={page_num}"

        log.info("Fetching index page %d: %s", page_num, url)
        resp = fetch(url)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Find post links - WordPress usually wraps titles in h2 > a inside article or post-list
        post_links = []

        # Strategy 1: Look for article elements with post links
        articles = soup.find_all("h2", class_=re.compile(r"wp-block-post-title|entry-title"))
        for h2 in articles:
            a = h2.find("a")
            if a and a.get("href"):
                post_links.append({
                    "url": a["href"],
                    "title": a.get_text(strip=True)
                })

        # Strategy 2: Look for post links in a list/grid
        if not post_links:
            # Try finding links within wp-block-post-template or similar
            for a in soup.select("li.wp-block-post a, .wp-block-query a"):
                href = a.get("href", "")
                title = a.get_text(strip=True)
                if href and title and href.startswith(BASE_URL):
                    post_links.append({"url": href, "title": title})

        # Strategy 3: Broader search
        if not post_links:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                title = a.get_text(strip=True)
                if (href.startswith(BASE_URL + "/")
                    and href != BASE_URL + "/"
                    and title
                    and len(title) > 15
                    and not any(skip in href for skip in ["/page/", "/category/", "/tag/", "/author/", "?query"])):
                    post_links.append({"url": href, "title": title})

        if not post_links:
            log.info("No posts found on page %d, stopping.", page_num)
            break

        new_count = 0
        for p in post_links:
            if p["url"] not in seen_urls:
                seen_urls.add(p["url"])
                all_posts.append(p)
                new_count += 1

        log.info("Found %d new posts on page %d (total: %d)", new_count, page_num, len(all_posts))

        if new_count == 0:
            log.info("No new posts on page %d, stopping.", page_num)
            break

        time.sleep(REQUEST_DELAY)

    return all_posts


# ---------------------------------------------------------------------------
# Process a single blog post
# ---------------------------------------------------------------------------

def process_post(post: dict, index: int, total: int):
    """Fetch a blog post, convert to markdown, save with images."""
    url = post["url"]
    title = post["title"]
    log.info("[%d/%d] Processing: %s", index, total, title)

    # Create output directory
    slug = slugify(title)
    if not slug:
        slug = f"post-{index}"
    post_dir = OUTPUT_DIR / slug
    img_dir = post_dir / "images"
    post_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the post
    resp = fetch(url)
    if resp is None:
        log.error("  Skipping: could not fetch %s", url)
        return

    soup = BeautifulSoup(resp.text, "lxml")

    # Extract and convert
    title_extracted, date_str, md_content = extract_post_content(soup, url, img_dir)

    # If title from page is better, use it
    if title_extracted:
        title = title_extracted

    # Build final markdown
    lines = []
    lines.append(f"# {title}\n")
    if date_str:
        lines.append(f"**Date:** {date_str}\n")
    lines.append(f"**Source:** [{url}]({url})\n")
    lines.append("---\n")
    lines.append(md_content)

    md_text = "\n".join(lines)

    # Write markdown file
    md_path = post_dir / "index.md"
    md_path.write_text(md_text, encoding="utf-8")
    log.info("  Saved: %s", md_path)

    # Clean up empty images directory
    if img_dir.exists() and not any(img_dir.iterdir()):
        img_dir.rmdir()

    time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Colfax Research Blog Scraper")
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover all blog post URLs
    log.info("Step 1: Discovering blog posts...")
    posts = get_all_post_urls()
    log.info("Found %d blog posts total.", len(posts))

    if not posts:
        log.error("No posts found. The site structure may have changed.")
        sys.exit(1)

    # Save post list for reference
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    log.info("Saved manifest to %s", manifest_path)

    # Step 2: Process each post
    log.info("Step 2: Processing blog posts...")
    for i, post in enumerate(posts, 1):
        try:
            process_post(post, i, len(posts))
        except Exception as e:
            log.error("[%d/%d] Error processing %s: %s", i, len(posts), post["url"], e)
            import traceback
            traceback.print_exc()

    log.info("=" * 60)
    log.info("Done! Blog posts saved to: %s", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
