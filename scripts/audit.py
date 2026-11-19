import sys

from crosstrace.audit import (
    NAMES,
    context,
    excerpt,
    fetch_issue,
    load_labels,
    next_unlabeled,
    sample_pairs,
    save_label,
    summarize,
)
from crosstrace.config import GITHUB_TOKEN
from crosstrace.db import connect
from crosstrace.github import GitHubClient
from crosstrace.projects import load_projects

# Raise this number later to label more pairs.
LIMIT = 100

if "summary" in sys.argv:
    print(summarize(load_labels()))
    sys.exit()

if not GITHUB_TOKEN:
    sys.exit("GITHUB_TOKEN is missing. Check your .env file.")

repos = {p.repo for p in load_projects()}
pairs = sample_pairs(connect(), repos)[:LIMIT]
client = GitHubClient(GITHUB_TOKEN)
labels = load_labels()


def show(tag: str, repo: str, number: int, mention_of: tuple[str, int] | None) -> None:
    issue = fetch_issue(client, repo, number)
    print(f"{tag}: {repo}#{number}")
    if issue is None:
        print("  (not available)")
        return
    kind = "PR" if "pull_request" in issue else "issue"
    print(f"  {kind}, {issue['state']}: {issue['title']}")
    print(f"  {issue['html_url']}")
    body = issue.get("body")
    if mention_of:
        snippet = context(body, *mention_of)
        if snippet:
            print(f"  MENTION: ...{snippet}...")
            return
        print("  (mention is not in the body: open the page)")
    print(f"  TEXT: {excerpt(body)}")


while True:
    pair = next_unlabeled(pairs, labels)
    if pair is None:
        print("All pairs in this sample are labeled.")
        break
    source_repo, source_number, target_repo, target_number = pair
    done = sum(1 for p in pairs if p in labels)
    print("=" * 72)
    print(f"[{done + 1}/{len(pairs)}]")
    show("SOURCE", source_repo, source_number, (target_repo, target_number))
    print()
    show("TARGET", target_repo, target_number, None)
    print()
    try:
        answer = input("c=cause r=related n=noise x=unclear q=quit > ").strip()
    except EOFError:
        break
    key = answer[:1].lower()
    if key == "q":
        break
    if key not in NAMES:
        print("Use c, r, n, x, or q.")
        continue
    note = answer[1:].strip()
    save_label(pair, NAMES[key], note)
    labels[pair] = NAMES[key]

print(summarize(labels))