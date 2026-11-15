from crosstrace.config import GITHUB_TOKEN
from crosstrace.github import GitHubClient

client = GitHubClient(GITHUB_TOKEN)
print(client.rate_limit())