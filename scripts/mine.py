import sys

from crosstrace.config import GITHUB_TOKEN
from crosstrace.db import connect
from crosstrace.github import GitHubClient
from crosstrace.mine import mine_repo
from crosstrace.projects import load_projects

# Newest issues per repository. Raise this number later.
LIMIT = 1000

if not GITHUB_TOKEN:
    sys.exit("GITHUB_TOKEN is missing. Check your .env file.")

repos = sys.argv[1:] or [p.repo for p in load_projects()]
client = GitHubClient(GITHUB_TOKEN)
conn = connect()
for repo in repos:
    mine_repo(client, conn, repo, limit=LIMIT)