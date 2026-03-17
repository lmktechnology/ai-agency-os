from __future__ import annotations

from typing import Any

import httpx


def web_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """
    Search the web using the DuckDuckGo Instant Answer API.
    Returns a list of results with title, url, and snippet.
    Note: For richer results, configure a SerpAPI or Brave Search key via env.
    """
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
                headers={"User-Agent": "ai-agency-os/1.0"},
            )
        data = response.json()
    except Exception as e:
        return {"error": str(e), "results": []}

    results = []

    # Abstract answer
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", "Abstract"),
            "url": data.get("AbstractURL", ""),
            "snippet": data["AbstractText"],
        })

    # Related topics
    for topic in data.get("RelatedTopics", [])[:num_results]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("Text", "")[:80],
                "url": topic.get("FirstURL", ""),
                "snippet": topic.get("Text", ""),
            })

    return {
        "query": query,
        "results": results[:num_results],
        "total": len(results),
    }
