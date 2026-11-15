import httpx

from crosstrace.github import GitHubClient


def make_client(handler, tmp_path, sleeps=None):
    record = sleeps.append if sleeps is not None else (lambda s: None)
    return GitHubClient(
        transport=httpx.MockTransport(handler),
        cache_dir=tmp_path,
        sleep=record,
    )


def test_paginate_follows_next_link(tmp_path):
    next_url = "https://api.github.com/x?per_page=100&page=2"

    def handler(request):
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[{"id": 3}])
        link = f'<{next_url}>; rel="next"'
        return httpx.Response(
            200, json=[{"id": 1}, {"id": 2}], headers={"link": link}
        )

    client = make_client(handler, tmp_path)
    ids = [item["id"] for item in client.paginate("/x")]
    assert ids == [1, 2, 3]


def test_cache_avoids_second_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, json=[{"id": 1}])

    client = make_client(handler, tmp_path)
    list(client.paginate("/x"))
    list(client.paginate("/x"))
    assert len(calls) == 1


def test_waits_when_rate_limited(tmp_path):
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "0"}
            return httpx.Response(403, headers=headers)
        return httpx.Response(200, json=[])

    sleeps = []
    client = make_client(handler, tmp_path, sleeps)
    list(client.paginate("/x"))
    assert len(sleeps) == 1