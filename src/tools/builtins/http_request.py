from __future__ import annotations

from typing import Any

import httpx


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Make an HTTP GET or POST request and return the response."""
    method = method.upper()
    with httpx.Client(timeout=timeout) as client:
        if method == "GET":
            response = client.get(url, headers=headers or {})
        elif method == "POST":
            response = client.post(url, json=body, headers=headers or {})
        else:
            return {"error": f"Unsupported method: {method}"}

    try:
        json_body = response.json()
    except Exception:
        json_body = None

    return {
        "status_code": response.status_code,
        "url": str(response.url),
        "json": json_body,
        "text": response.text[:2000] if json_body is None else None,
    }
