from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import httpx


SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "local-ai-lab/0.1 (+https://github.com/Mofang47/ai_models)"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    excerpt: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class DuckDuckGoResultsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._current_href: str | None = None
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._capture_title = False
        self._capture_snippet = False
        self._pending_result: SearchResult | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "") or ""

        if tag == "a" and "result__a" in classes and attrs_dict.get("href"):
            self._current_href = attrs_dict["href"]
            self._current_title = []
            self._capture_title = True
            return

        if tag in {"a", "div"} and "result__snippet" in classes:
            self._current_snippet = []
            self._capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title and self._current_href:
            title = normalize_whitespace("".join(self._current_title))
            url = normalize_duckduckgo_url(self._current_href)
            if title and url:
                self._pending_result = SearchResult(title=title, url=url, snippet="")
            self._capture_title = False
            self._current_href = None
            self._current_title = []
            return

        if tag in {"a", "div"} and self._capture_snippet:
            snippet = normalize_whitespace("".join(self._current_snippet))
            if self._pending_result and snippet:
                self._pending_result.snippet = snippet
                self.results.append(self._pending_result)
                self._pending_result = None
            self._capture_snippet = False
            self._current_snippet = []

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title.append(data)
        if self._capture_snippet:
            self._current_snippet.append(data)

    def close(self) -> None:
        super().close()
        if self._pending_result:
            self.results.append(self._pending_result)
            self._pending_result = None


class HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignore_depth += 1
        elif tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignore_depth:
            self._ignore_depth -= 1
        elif tag in {"p", "div", "section", "article", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignore_depth:
            self.parts.append(data)

    def get_text(self) -> str:
        return normalize_whitespace(" ".join(self.parts))


def normalize_whitespace(value: str) -> str:
    return " ".join(unescape(value).split())


def normalize_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        return uddg
    if url.startswith("//"):
        return f"https:{url}"
    return url


class WebSearchClient:
    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(SEARCH_URL, params={"q": query})
            response.raise_for_status()

        parser = DuckDuckGoResultsParser()
        parser.feed(response.text)
        parser.close()

        deduped: list[SearchResult] = []
        seen: set[str] = set()
        for result in parser.results:
            if result.url in seen or not result.url.startswith(("http://", "https://")):
                continue
            deduped.append(result)
            seen.add(result.url)
            if len(deduped) >= max_results:
                break
        return deduped

    async def fetch_excerpt(self, url: str, max_chars: int = 1400) -> str:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/plain" in content_type:
            text = response.text
        else:
            parser = HTMLTextParser()
            parser.feed(response.text)
            parser.close()
            text = parser.get_text()

        return text[:max_chars].strip()

    async def research(self, query: str, max_results: int = 4, excerpt_chars: int = 1200) -> list[SearchResult]:
        results = await self.search(query=query, max_results=max_results)
        excerpts = await asyncio.gather(
            *(self.fetch_excerpt(result.url, max_chars=excerpt_chars) for result in results),
            return_exceptions=True,
        )

        enriched: list[SearchResult] = []
        for result, excerpt in zip(results, excerpts, strict=False):
            if isinstance(excerpt, Exception):
                result.excerpt = result.snippet
            else:
                result.excerpt = excerpt or result.snippet
            enriched.append(result)
        return enriched

    async def build_grounding(
        self,
        query: str,
        max_results: int = 4,
        excerpt_chars: int = 1200,
    ) -> tuple[str, list[dict[str, Any]]]:
        results = await self.research(query=query, max_results=max_results, excerpt_chars=excerpt_chars)
        if not results:
            return "", []

        sections: list[str] = []
        sources: list[dict[str, Any]] = []
        for index, result in enumerate(results, start=1):
            sections.append(
                "\n".join(
                    [
                        f"[{index}] {result.title}",
                        f"URL: {result.url}",
                        f"Search snippet: {result.snippet}",
                        f"Page excerpt: {result.excerpt}",
                    ]
                )
            )
            sources.append(result.to_dict())

        context = (
            "Use the following web research results when answering if they are relevant. "
            "Cite supporting sources inline like [1] or [2]. If the sources are incomplete "
            "or conflicting, say that clearly.\n\n"
            + "\n\n".join(sections)
        )
        return context, sources
