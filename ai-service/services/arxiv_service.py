import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
ARXIV_ID_RE = re.compile(r"^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$|^[a-z-]+(\.[A-Z]{2})?/[0-9]{7}(v[0-9]+)?$")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DOWNLOAD_TIMEOUT_SECONDS = 30


class ArxivServiceError(Exception):
    pass


def _text(element: ElementTree.Element | None) -> str:
    return " ".join((element.text or "").split()) if element is not None else ""


def _arxiv_id_from_abs_url(abs_url: str) -> str:
    path = urlparse(abs_url).path.rstrip("/")
    if "/abs/" in path:
        return path.rsplit("/abs/", 1)[-1]
    return path.rsplit("/", 1)[-1]


def _https_arxiv_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def _pdf_url_from_entry(entry: ElementTree.Element, arxiv_id: str) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            href = link.attrib.get("href", "")
            if href.startswith("http://arxiv.org/"):
                return href.replace("http://arxiv.org/", "https://arxiv.org/", 1)
            return href
    return f"https://arxiv.org/pdf/{arxiv_id}"


def is_valid_arxiv_pdf_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme != "https":
        return False
    if parsed.netloc.lower() not in ARXIV_HOSTS:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] != "pdf":
        return False
    arxiv_id = parts[1].removesuffix(".pdf")
    return bool(ARXIV_ID_RE.match(arxiv_id))


def safe_arxiv_pdf_filename(arxiv_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(arxiv_id or "arxiv-paper")).strip(".-")
    return f"arxiv-{cleaned or 'paper'}.pdf"


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    cleaned_query = " ".join(str(query or "").split())
    if not cleaned_query:
        raise ArxivServiceError("Enter a search query.")

    bounded_max_results = max(1, min(int(max_results or 10), 25))
    params = urlencode({
        "search_query": f"all:{cleaned_query}",
        "start": 0,
        "max_results": bounded_max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    request = Request(
        f"{ARXIV_API_URL}?{params}",
        headers={"User-Agent": "DeepDoc/0.1 (research paper analysis tool)"},
    )

    try:
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        raise ArxivServiceError("Could not search arXiv right now.") from error

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise ArxivServiceError("arXiv returned an unreadable search response.") from error

    results: list[dict] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_abs_url = _text(entry.find("atom:id", ATOM_NS))
        arxiv_id = _arxiv_id_from_abs_url(entry_abs_url)
        authors = [
            _text(author.find("atom:name", ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]
        results.append({
            "title": _text(entry.find("atom:title", ATOM_NS)),
            "authors": [author for author in authors if author],
            "summary": _text(entry.find("atom:summary", ATOM_NS)),
            "published": _text(entry.find("atom:published", ATOM_NS))[:10],
            "updated": _text(entry.find("atom:updated", ATOM_NS))[:10],
            "arxiv_id": arxiv_id,
            "abs_url": _https_arxiv_abs_url(arxiv_id),
            "pdf_url": _pdf_url_from_entry(entry, arxiv_id),
            "categories": categories,
        })

    return results


def download_arxiv_pdf(pdf_url: str, arxiv_id: str, target_dir: str | Path | None = None) -> Path:
    if not is_valid_arxiv_pdf_url(pdf_url):
        raise ArxivServiceError("Invalid arXiv PDF URL.")

    output_dir = Path(target_dir) if target_dir is not None else Path(tempfile.gettempdir())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_arxiv_pdf_filename(arxiv_id)
    request = Request(
        pdf_url,
        headers={"User-Agent": "DeepDoc/0.1 (research paper analysis tool)"},
    )

    try:
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            content = response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        raise ArxivServiceError("Could not download the arXiv PDF.") from error

    if not content.startswith(b"%PDF"):
        raise ArxivServiceError("arXiv did not return a valid PDF file.")

    output_path.write_bytes(content)
    return output_path
