import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
ARXIV_ID_RE = re.compile(r"^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$|^[a-z-]+(\.[A-Z]{2})?/[0-9]{7}(v[0-9]+)?$")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
OPENSEARCH_NS = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
DOWNLOAD_TIMEOUT_SECONDS = 30
SEARCH_RETRY_DELAY_SECONDS = 0.75
SEARCH_RETRY_HTTP_CODES = {400, 408, 429, 500, 502, 503, 504}
SEARCH_FIELD_PREFIXES = {
    "all": "all",
    "title": "ti",
    "author": "au",
    "abstract": "abs",
}
SORT_FIELDS = {"relevance", "lastUpdatedDate", "submittedDate"}
SORT_ORDERS = {"ascending", "descending"}
CATEGORY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ArxivServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


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
    return arxiv_id_from_pdf_url(url) is not None


def arxiv_id_from_pdf_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme != "https":
        return None
    if parsed.netloc.lower() not in ARXIV_HOSTS:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] != "pdf":
        return None
    arxiv_id = "/".join(parts[1:]).removesuffix(".pdf")
    return arxiv_id if ARXIV_ID_RE.match(arxiv_id) else None


def safe_arxiv_pdf_filename(arxiv_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(arxiv_id or "arxiv-paper")).strip(".-")
    return f"arxiv-{cleaned or 'paper'}.pdf"


def _int_text(element: ElementTree.Element | None, default: int = 0) -> int:
    try:
        return int(_text(element))
    except ValueError:
        return default


def _bounded_int(value: object, default: int, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(value if value is not None and value != "" else default)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _normalize_search_field(search_field: str | None) -> str:
    return str(search_field or "all").strip().lower() or "all"


def _normalize_sort_by(sort_by: str | None) -> str:
    value = str(sort_by or "relevance").strip()
    return value if value in SORT_FIELDS else "relevance"


def _normalize_sort_order(sort_order: str | None) -> str:
    value = str(sort_order or "descending").strip()
    return value if value in SORT_ORDERS else "descending"


def _normalize_category(category: str | None) -> str:
    value = str(category or "").strip()
    if not value or value.lower() == "all":
        return ""
    if not CATEGORY_RE.match(value):
        raise ArxivServiceError("Invalid arXiv category filter.")
    return value


def _read_search_response(request: Request) -> bytes:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                return response.read()
        except HTTPError as error:
            last_error = error
            if error.code not in SEARCH_RETRY_HTTP_CODES or attempt == 1:
                break
        except (URLError, TimeoutError) as error:
            last_error = error
            if attempt == 1:
                break

        time.sleep(SEARCH_RETRY_DELAY_SECONDS)

    if isinstance(last_error, HTTPError):
        if last_error.code == 429:
            message = "arXiv is rate limiting search right now. Try again in a moment."
        else:
            message = "arXiv search is temporarily unavailable. Try again or use a more specific query."
        raise ArxivServiceError(message, status_code=502) from last_error

    raise ArxivServiceError("Could not reach arXiv right now.", status_code=502) from last_error


def _build_search_query(query: str, search_field: str, category: str) -> str:
    prefix = SEARCH_FIELD_PREFIXES.get(search_field, "all")
    search_query = f"{prefix}:{query}"
    if category:
        search_query = f"{search_query} AND cat:{category}"
    return search_query


def search_arxiv_page(
    query: str,
    max_results: int = 20,
    start: int = 0,
    search_field: str = "all",
    category: str = "",
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> dict:
    cleaned_query = " ".join(str(query or "").split())
    if not cleaned_query:
        raise ArxivServiceError("Enter a search query.")

    bounded_max_results = _bounded_int(max_results, default=20, minimum=1, maximum=25)
    bounded_start = _bounded_int(start, default=0, minimum=0)
    normalized_search_field = _normalize_search_field(search_field)
    normalized_category = _normalize_category(category)
    normalized_sort_by = _normalize_sort_by(sort_by)
    normalized_sort_order = _normalize_sort_order(sort_order)
    params = urlencode({
        "search_query": _build_search_query(cleaned_query, normalized_search_field, normalized_category),
        "start": bounded_start,
        "max_results": bounded_max_results,
        "sortBy": normalized_sort_by,
        "sortOrder": normalized_sort_order,
    })
    request = Request(
        f"{ARXIV_API_URL}?{params}",
        headers={"User-Agent": "DeepDoc/0.1 (research paper analysis tool)"},
    )

    payload = _read_search_response(request)

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise ArxivServiceError("arXiv returned an unreadable search response.", status_code=502) from error

    total_results = _int_text(root.find("opensearch:totalResults", OPENSEARCH_NS))
    start_index = _int_text(root.find("opensearch:startIndex", OPENSEARCH_NS), bounded_start)
    items_per_page = _int_text(root.find("opensearch:itemsPerPage", OPENSEARCH_NS), bounded_max_results)

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

    page_size = items_per_page or bounded_max_results
    page = start_index // page_size + 1 if page_size else 1
    total_pages = (total_results + page_size - 1) // page_size if page_size and total_results else 0
    return {
        "results": results,
        "query": cleaned_query,
        "search_field": normalized_search_field,
        "category": normalized_category,
        "sort_by": normalized_sort_by,
        "sort_order": normalized_sort_order,
        "start": start_index,
        "max_results": page_size,
        "total_results": total_results,
        "page": page,
        "total_pages": total_pages,
    }


def search_arxiv(
    query: str,
    max_results: int = 20,
    start: int = 0,
    search_field: str = "all",
    category: str = "",
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> list[dict]:
    return search_arxiv_page(
        query,
        max_results=max_results,
        start=start,
        search_field=search_field,
        category=category,
        sort_by=sort_by,
        sort_order=sort_order,
    )["results"]


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
        raise ArxivServiceError("Could not download the arXiv PDF.", status_code=502) from error

    if not content.startswith(b"%PDF"):
        raise ArxivServiceError("arXiv did not return a valid PDF file.", status_code=502)

    output_path.write_bytes(content)
    return output_path
