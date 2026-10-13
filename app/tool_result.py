"""
What a tool returns to the agent loop.

A tool gives the model text (content), but it can also hand structured data to
the application that the model never has to repeat -- today the web pages a
search found (sources). The agent collects them and returns them in the API
response, so the caller gets reliable links instead of hoping the model lists
them correctly.

Tools may still return a plain str; the agent treats it as ToolResult(content=...).
"""

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class ToolResult:
    content: str
    sources: tuple[Source, ...] = ()


def make_source(title: object, url: object) -> Source | None:
    """
    Builds a Source from untrusted data (e.g. a search result), or None if the URL
    is not a plain http(s) link. The caller renders sources as clickable links, so
    a "javascript:" or "data:" URL must never get through.
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    title_text = " ".join(str(title or "").split()) or url
    return Source(title=title_text, url=url)
