import httpx

from crosstrace.db import connect
from crosstrace.github import GitHubClient
from crosstrace.mine import mine_repo

RESPONSE = {
    "data": {
        "rateLimit": {"cost": 1, "remaining": 4999},
        "repository": {
            "issues": {
                "pageInfo": {"hasNextPage": False, "endCursor": None},
                "nodes": [
                    {
                        "number": 10,
                        "title": "Bug in foo",
                        "body": "text",
                        "state": "CLOSED",
                        "createdAt": "2024-01-01T00:00:00Z",
                        "closedAt": "2024-02-01T00:00:00Z",
                        "author": {"login": "ann"},
                        "timelineItems": {
                            "pageInfo": {"hasNextPage": False},
                            "nodes": [
                                {
                                    "createdAt": "2024-01-05T00:00:00Z",
                                    "isCrossRepository": True,
                                    "willCloseTarget": False,
                                    "source": {
                                        "__typename": "Issue",
                                        "number": 5,
                                        "repository": {"nameWithOwner": "c/d"},
                                    },
                                },
                                {
                                    "createdAt": "2024-01-06T00:00:00Z",
                                    "isCrossRepository": False,
                                    "willCloseTarget": False,
                                    "source": {
                                        "__typename": "Issue",
                                        "number": 6,
                                        "repository": {"nameWithOwner": "a/b"},
                                    },
                                },
                            ],
                        },
                    }
                ],
            }
        },
    }
}


def test_mine_repo_saves_only_cross_repo_links(tmp_path):
    def handler(request):
        return httpx.Response(200, json=RESPONSE)

    client = GitHubClient(
        transport=httpx.MockTransport(handler),
        cache_dir=tmp_path / "raw",
        sleep=lambda s: None,
    )
    conn = connect(tmp_path / "test.sqlite")

    assert mine_repo(client, conn, "a/b") == 1

    issues = conn.execute("SELECT repo, number, author FROM issues").fetchall()
    assert issues == [("a/b", 10, "ann")]

    refs = conn.execute(
        "SELECT source_repo, source_number, target_repo, target_number "
        "FROM cross_refs"
    ).fetchall()
    assert refs == [("c/d", 5, "a/b", 10)]