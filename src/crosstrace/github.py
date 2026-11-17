import hashlib
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

from crosstrace.config import RAW_DIR

API = "https://api.github.com"


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        cache_dir: Path = RAW_DIR,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            base_url=API, headers=headers, timeout=30, transport=transport
        )
        self._cache_dir = Path(cache_dir)
        self._sleep = sleep

    def _request(
        self, url: str, params: dict | None = None, body: dict | None = None
    ) -> httpx.Response:
        while True:
            if body is None:
                response = self._http.get(url, params=params)
            else:
                response = self._http.post(url, json=body)
            if "retry-after" in response.headers:
                self._sleep(int(response.headers["retry-after"]) + 1)
                continue
            limited = response.headers.get("x-ratelimit-remaining") == "0"
            if response.status_code in (403, 429) and limited:
                reset = int(response.headers["x-ratelimit-reset"])
                self._sleep(max(reset - time.time(), 0) + 1)
                continue
            response.raise_for_status()
            return response

    def _cache_path(self, url: str, params: dict | None) -> Path:
        key = f"{url}?{sorted((params or {}).items())}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def get(self, url: str, params: dict | None = None) -> dict:
        path = self._cache_path(url, params)
        if path.exists():
            return json.loads(path.read_text())
        response = self._request(url, params)
        page = {
            "url": url,
            "params": params,
            "data": response.json(),
            "next": response.links.get("next", {}).get("url"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(page))
        return page

    def paginate(self, url: str, params: dict | None = None) -> Iterator[dict]:
        params = {"per_page": 100, **(params or {})}
        while url:
            page = self.get(url, params)
            yield from page["data"]
            url, params = page["next"], None

    def rate_limit(self) -> dict:
        return self._request("/rate_limit", None).json()["resources"]["core"]

    def graphql(self, query: str, variables: dict) -> dict:
        key = {"query": query, "variables": json.dumps(variables, sort_keys=True)}
        path = self._cache_path("/graphql", key)
        if path.exists():
            return json.loads(path.read_text())["data"]
        response = self._request(
            "/graphql", body={"query": query, "variables": variables}
        )
        result = response.json()
        if "errors" in result:
            raise RuntimeError(result["errors"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        return result["data"]