import sqlite3

from crosstrace.github import GitHubClient

QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  rateLimit { cost remaining }
  repository(owner: $owner, name: $name) {
    issues(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        state
        createdAt
        closedAt
        author { login }
        timelineItems(first: 100, itemTypes: [CROSS_REFERENCED_EVENT]) {
          pageInfo { hasNextPage }
          nodes {
            ... on CrossReferencedEvent {
              createdAt
              isCrossRepository
              willCloseTarget
              source {
                __typename
                ... on Issue { number repository { nameWithOwner } }
                ... on PullRequest { number repository { nameWithOwner } }
              }
            }
          }
        }
      }
    }
  }
}
"""


def save_issue(conn: sqlite3.Connection, repo: str, node: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO issues VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            repo,
            node["number"],
            node["title"],
            node["body"],
            node["state"],
            (node["author"] or {}).get("login"),
            node["createdAt"],
            node["closedAt"],
        ),
    )


def save_cross_ref(
    conn: sqlite3.Connection, repo: str, number: int, event: dict
) -> None:
    source = event.get("source")
    if not source or not event.get("isCrossRepository"):
        return
    conn.execute(
        "INSERT OR REPLACE INTO cross_refs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            source["repository"]["nameWithOwner"],
            source["number"],
            source["__typename"],
            repo,
            number,
            event["createdAt"],
            int(event["willCloseTarget"]),
        ),
    )

def mine_repo(
    client: GitHubClient,
    conn: sqlite3.Connection,
    repo: str,
    limit: int | None = None,
) -> int:
    owner, name = repo.split("/")
    cursor = None
    count = 0
    truncated = 0
    while True:
        data = client.graphql(
            QUERY, {"owner": owner, "name": name, "cursor": cursor}
        )
        issues = data["repository"]["issues"]
        for node in issues["nodes"]:
            save_issue(conn, repo, node)
            timeline = node["timelineItems"]
            if timeline["pageInfo"]["hasNextPage"]:
                truncated += 1
            for event in timeline["nodes"]:
                save_cross_ref(conn, repo, node["number"], event)
            count += 1
        conn.commit()
        remaining = data.get("rateLimit", {}).get("remaining")
        print(f"{repo}: {count} issues, API points left: {remaining}")
        if not issues["pageInfo"]["hasNextPage"]:
            break
        if limit and count >= limit:
            break
        cursor = issues["pageInfo"]["endCursor"]
    if truncated:
        print(f"{repo}: WARNING {truncated} issues have more than 100 events")
    return count