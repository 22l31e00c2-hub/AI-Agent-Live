"""Best-effort web reading for amu.

No API key, no paid service, no headless browser: one HTTP GET through the
standard library, a byte limit, a timeout, an SSRF guard, and a deterministic
keyword summariser. It reads pages; it cannot execute their JavaScript or log
in anywhere.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

MAX_BYTES = 2_000_000
DEFAULT_TIMEOUT = 15
USER_AGENT = "amu-cadagent/0.1 (local; no external API)"
CACHE_TTL_SECONDS = 6 * 60 * 60

SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
WIKI_ENDPOINT = "https://en.wikipedia.org/w/api.php"

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
ANCHOR_RE = re.compile(r"(?is)<a\b([^>]*)>(.*?)</a>")
SEARCH_INTENT_RE = re.compile(
    r"\b(?:search\s+(?:the\s+)?web|search\s+online|web\s+search|google)\b", re.I
)

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about",
    "your", "you", "are", "was", "were", "have", "has", "how", "what", "when",
    "which", "will", "can", "not", "but", "all", "any", "out", "get", "use",
    "using", "used", "its", "it's", "then", "than", "them", "they", "more",
    "most", "some", "such", "only", "also", "over", "under", "per",
}


class WebError(Exception):
    """Any reason a page could not be read."""


@dataclass
class Page:
    url: str
    status: int
    content_type: str
    text: str
    title: str = ""
    cached: bool = False
    raw: str = ""
    age_seconds: Optional[float] = None
    stale: bool = False
    note: str = ""


@dataclass
class Lookup:
    page: Page
    sentences: List[str] = field(default_factory=list)


@dataclass
class SearchOutcome:
    query: str
    provider: str
    results: List[Tuple[str, str]] = field(default_factory=list)


# ----------------------------------------------------------------------
# request parsing
# ----------------------------------------------------------------------
def find_urls(text: str) -> List[str]:
    urls: List[str] = []
    for match in URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?)]}'\"")
        if url and url not in urls:
            urls.append(url)
    return urls


def is_search_request(text: str) -> bool:
    """True for an explicit 'search the web' style request."""
    return bool(SEARCH_INTENT_RE.search(text)) and not find_urls(text)


def search_query(request: str) -> str:
    query = SEARCH_INTENT_RE.sub(" ", request, count=1)
    query = re.sub(r"\b(for|about|on|online|the web)\b", " ", query, flags=re.I)
    query = re.sub(r"\s+", " ", query).strip(" ?.,:")
    return query or request.strip()


# ----------------------------------------------------------------------
# fetching
# ----------------------------------------------------------------------
def _as_ip(host: str) -> Optional[ipaddress._BaseAddress]:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_blocked(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def guard_url(url: str, resolve: bool = True) -> str:
    """Reject anything that is not a public http(s) target (SSRF guard)."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise WebError("only http and https URLs are allowed, not %r" % parsed.scheme)

    host = parsed.hostname or ""
    if not host:
        raise WebError("URL has no host: %s" % url)
    if host.lower() in ("localhost", "localhost.localdomain") or host.lower().endswith(".local"):
        raise WebError("refusing to fetch a local address: %s" % host)

    literal = _as_ip(host)
    if literal is not None and _is_blocked(literal):
        raise WebError("refusing to fetch a private address: %s" % host)

    # A hostname can resolve to a private address; check what it really points
    # at. (DNS can still change between this check and the request - acceptable
    # for a local tool, not for a shared server.)
    if resolve and literal is None:
        try:
            addresses = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise WebError("cannot resolve %s: %s" % (host, exc))
        for info in addresses:
            ip = _as_ip(info[4][0])
            if ip is not None and _is_blocked(ip):
                raise WebError("refusing to fetch a private address: %s -> %s" % (host, ip))

    return url


def _urllib_transport(url: str, timeout: int, max_bytes: int) -> Tuple[int, Dict[str, str], bytes]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(max_bytes + 1)
        return response.status, dict(response.headers), body


def _cache_file(url: str, cache_dir) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / ("%s.json" % digest)


def _load_cache(url: str, cache_dir) -> Optional[Tuple[Page, Optional[float]]]:
    path = _cache_file(url, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("url") != url:
        return None

    age: Optional[float] = None
    stamp = data.get("fetched_ts")
    if isinstance(stamp, (int, float)):
        age = max(0.0, time.time() - float(stamp))
    else:
        written = data.get("fetched_at") or ""
        try:
            age = max(0.0, time.time() - time.mktime(time.strptime(written, "%Y-%m-%dT%H:%M:%S")))
        except (ValueError, OverflowError):
            age = None

    page = Page(
        url=url,
        status=int(data.get("status", 200)),
        content_type=data.get("content_type", ""),
        text=data.get("text", ""),
        title=data.get("title", ""),
        cached=True,
        raw=data.get("raw", ""),
        age_seconds=age,
    )
    return page, age


def _save_cache(page: Page, cache_dir) -> None:
    path = _cache_file(page.url, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "url": page.url,
                    "status": page.status,
                    "content_type": page.content_type,
                    "text": page.text[:200_000],
                    "title": page.title,
                    "raw": page.raw[:200_000],
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "fetched_ts": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _decode(body: bytes, charset: str = "utf-8") -> str:
    """Decode a body, ignoring a wrong declared charset.

    Documentation sites often send a charset that does not match the bytes they
    send. Rather than print mojibake, keep the candidate decoding with the
    fewest replacement characters.
    """
    best: Optional[str] = None
    best_score: Optional[int] = None
    for candidate in (charset, "utf-8", "cp1252", "latin-1"):
        try:
            text = body.decode(candidate, errors="replace")
        except LookupError:
            continue
        score = text.count("\ufffd")
        if best_score is None or score < best_score:
            best, best_score = text, score
        if score == 0:
            break
    return best if best is not None else body.decode("utf-8", errors="replace")


def fetch(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
    transport: Optional[Callable[[str, int], Tuple[int, Dict[str, str], bytes]]] = None,
    cache_dir=None,
    max_age: float = CACHE_TTL_SECONDS,
) -> Page:
    """GET one URL and return readable text. Raises WebError on failure.

    `max_age` is how long a cached copy still counts as current, in seconds.
    Zero or less always refetches. When the network fails but an older copy
    exists, that copy comes back with `stale=True` and the reason in `note`, so
    the answer can say where it really came from.
    """
    guard_url(url, resolve=transport is None)

    stale_page: Optional[Page] = None
    if cache_dir:
        hit = _load_cache(url, cache_dir)
        if hit is not None:
            cached, age = hit
            if max_age > 0 and (age is None or age <= max_age):
                return cached
            stale_page = cached

    if transport is None:
        def transport(target: str, seconds: int):
            return _urllib_transport(target, seconds, max_bytes)

    try:
        status, headers, body = transport(url, timeout)
    except urllib.error.HTTPError as exc:
        if stale_page is not None:
            stale_page.stale = True
            stale_page.note = "HTTP %s" % exc.code
            return stale_page
        raise WebError("HTTP %s for %s" % (exc.code, url))
    except (WebError, urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        if stale_page is not None:
            stale_page.stale = True
            stale_page.note = str(exc)
            return stale_page
        if isinstance(exc, WebError):
            raise
        raise WebError("could not fetch %s: %s" % (url, exc))

    if len(body) > max_bytes:
        body = body[:max_bytes]

    headers = {str(key): str(value) for key, value in (headers or {}).items()}
    content_type = (
        headers.get("Content-Type") or headers.get("content-type") or ""
    )
    charset = "utf-8"
    match = re.search(r"charset=([\w\-]+)", content_type, re.I)
    if match:
        charset = match.group(1)

    raw = _decode(body, charset)
    is_html = "html" in content_type.lower() or "<html" in raw[:2000].lower()
    text = readable_text(raw) if is_html else raw.strip()

    page = Page(
        url=url,
        status=int(status),
        content_type=content_type,
        text=text,
        title=title_of(raw),
        raw=raw,
    )
    if cache_dir:
        _save_cache(page, cache_dir)
    return page


def lookup(
    url: str,
    question: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    transport=None,
    cache_dir=None,
    max_age: Optional[float] = None,
) -> Lookup:
    page = fetch(
        url,
        timeout=timeout,
        transport=transport,
        cache_dir=cache_dir,
        max_age=CACHE_TTL_SECONDS if max_age is None else max_age,
    )
    return Lookup(page=page, sentences=summarize(page.text, question))


def cache_note(page: Page) -> str:
    """Where a page came from, in words worth printing."""
    if not page.cached:
        return ""
    if page.stale:
        suffix = ": %s" % page.note if page.note else ""
        return "stale cache, the network failed%s" % suffix
    if page.age_seconds is None:
        return "cached"
    age = page.age_seconds
    if age < 3600:
        return "cached %d min ago" % int(age / 60)
    if age < 86400:
        return "cached %d h ago" % int(age / 3600)
    return "cached %d d ago" % int(age / 86400)


# ----------------------------------------------------------------------
# turning markup into text
# ----------------------------------------------------------------------
def title_of(markup: str) -> str:
    match = TITLE_RE.search(markup or "")
    if not match:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))).strip()


def readable_text(markup: str) -> str:
    """Strip scripts, styles and tags, keep block structure as line breaks."""
    text = re.sub(r"(?is)<(script|style|noscript|template|svg)\b.*?</\1>", " ", markup or "")
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>|</tr>|</table>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")

    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _terms(text: str) -> List[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(word) > 2 and word not in STOPWORDS
    ]


def summarize(text: str, question: Optional[str] = None, limit: int = 5) -> List[str]:
    """Pick the sentences that answer the question, deterministically."""
    flat = re.sub(r"\s+", " ", text or "")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", flat)]
    sentences = [sentence for sentence in sentences if len(sentence) > 40]
    if not sentences:
        sentences = [line.strip() for line in (text or "").splitlines() if len(line.strip()) > 40]
    if not sentences:
        return []

    wanted = set(_terms(question or ""))
    if not wanted:
        return sentences[:limit]

    scored = []
    for index, sentence in enumerate(sentences):
        overlap = len(wanted & set(_terms(sentence)))
        if overlap:
            scored.append((overlap, -index, sentence))

    if not scored:
        return sentences[:limit]

    scored.sort(reverse=True)
    best = scored[:limit]
    best.sort(key=lambda item: -item[1])
    return [sentence for _, _, sentence in best]


# ----------------------------------------------------------------------
# search (best effort, no API key)
# ----------------------------------------------------------------------
def _attr(attrs: str, name: str) -> str:
    match = re.search(r"\b" + name + r"\s*=\s*[\"']([^\"']*)[\"']", attrs, re.I)
    return match.group(1) if match else ""


def _unwrap_redirect(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlsplit(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def parse_search_results(markup: str, limit: int = 8) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for attrs, label in ANCHOR_RE.findall(markup or ""):
        if "result__a" not in attrs and "result-link" not in attrs:
            continue
        href = _attr(attrs, "href")
        if not href:
            continue
        url = _unwrap_redirect(html.unescape(href))
        title = html.unescape(re.sub(r"<[^>]+>", "", label)).strip()
        title = re.sub(r"\s+", " ", title)
        if not title or not url:
            continue
        if any(url == existing for _, existing in results):
            continue
        results.append((title, url))
        if len(results) >= limit:
            break
    return results


def _search_ddg_html(query, limit, timeout, transport):
    url = SEARCH_ENDPOINT + "?q=" + urllib.parse.quote_plus(query)
    page = fetch(url, timeout=timeout, transport=transport)
    return parse_search_results(page.raw, limit=limit)


def _search_ddg_lite(query, limit, timeout, transport):
    url = LITE_ENDPOINT + "?q=" + urllib.parse.quote_plus(query)
    page = fetch(url, timeout=timeout, transport=transport)
    return parse_search_results(page.raw, limit=limit)


def _search_wikipedia(query, limit, timeout, transport):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": str(limit),
    }
    page = fetch(
        WIKI_ENDPOINT + "?" + urllib.parse.urlencode(params),
        timeout=timeout,
        transport=transport,
    )
    try:
        data = json.loads(page.raw)
    except ValueError:
        raise WebError("wikipedia returned non-JSON")

    results: List[Tuple[str, str]] = []
    for hit in (data.get("query", {}).get("search") or []):
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        snippet = html.unescape(re.sub(r"<[^>]+>", "", hit.get("snippet") or ""))
        snippet = re.sub(r"\s+", " ", snippet).strip()
        label = title if not snippet else "%s - %s" % (title, snippet[:140])
        results.append(
            (
                label,
                "https://en.wikipedia.org/wiki/"
                + urllib.parse.quote(title.replace(" ", "_")),
            )
        )
    return results


PROVIDERS = (
    ("duckduckgo-html", _search_ddg_html),
    ("duckduckgo-lite", _search_ddg_lite),
    ("wikipedia", _search_wikipedia),
)


def search(
    query: str,
    limit: int = 8,
    timeout: int = DEFAULT_TIMEOUT,
    transport=None,
) -> SearchOutcome:
    """Try each keyless provider in turn; return the first with results.

    There is no search API key here, so this is best effort: DuckDuckGo's two
    endpoints can be rate limited or blocked, and Wikipedia only covers
    encyclopaedic topics. When every provider fails you get a WebError; then
    `look up <url>` with a page you already know is the reliable path.
    """
    failures: List[str] = []

    for name, provider in PROVIDERS:
        try:
            results = provider(query, limit, timeout, transport)
        except WebError as exc:
            failures.append("%s: %s" % (name, exc))
            continue
        if results:
            return SearchOutcome(query=query, provider=name, results=results[:limit])
        failures.append("%s: no results parsed" % name)

    raise WebError("; ".join(failures) or "no provider returned results")
