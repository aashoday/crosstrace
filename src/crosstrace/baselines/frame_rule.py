"""Traceback-frame rule baseline.

Rule: take the packages from the frames of the query traceback.
Map them to repositories. Keep the candidates of those repositories that
pass the time rule. Sort by date, newest first.
A query with no upstream frame gets an empty list.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from crosstrace.benchmark import Candidate, Query, eligible
from crosstrace.traceback_parser import parse_tracebacks

# Assumption: check these names against projects.toml.
DEFAULT_PACKAGE_REPOS: dict[str, str] = {
    "numpy": "numpy/numpy",
    "scipy": "scipy/scipy",
    "pandas": "pandas-dev/pandas",
    "matplotlib": "matplotlib/matplotlib",
    "scikit-learn": "scikit-learn/scikit-learn",
    "scikit-image": "scikit-image/scikit-image",
    "astropy": "astropy/astropy",
    "xarray": "pydata/xarray",
    "statsmodels": "statsmodels/statsmodels",
    "networkx": "networkx/networkx",
    "dask": "dask/dask",
    "aeon": "aeon-toolkit/aeon",
}


class FrameRule:
    """Rank candidates by the packages in the query traceback."""

    def __init__(
        self,
        candidates: Iterable[Candidate],
        package_repos: Mapping[str, str] | None = None,
    ) -> None:
        self._package_repos = dict(package_repos or DEFAULT_PACKAGE_REPOS)
        self._by_repo: dict[str, list[Candidate]] = defaultdict(list)
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.id in seen:
                raise ValueError("candidate IDs must be unique")
            seen.add(candidate.id)
            self._by_repo[candidate.repo].append(candidate)

    def upstream_repos(self, query: Query) -> list[str]:
        """Repositories of the frame packages. Not the query's own one."""
        parsed = parse_tracebacks(f"{query.title}\n{query.body}")
        repos: list[str] = []
        for traceback in parsed:
            for package in traceback.packages:
                repo = self._package_repos.get(package)
                if repo and repo != query.repo and repo not in repos:
                    repos.append(repo)
        return repos

    def search(
        self,
        query: Query,
        exclude_same_repo: bool = True,
        top_k: int = 100,
    ) -> list[str]:
        """Return candidate IDs, newest first."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        pool: list[Candidate] = []
        for repo in self.upstream_repos(query):
            pool.extend(self._by_repo.get(repo, []))
        allowed = eligible(query, pool, exclude_same_repo)
        ranked = sorted(allowed, key=lambda c: (-c.date.timestamp(), c.id))
        return [c.id for c in ranked[:top_k]]

    def count_without_upstream_frame(self, queries: Iterable[Query]) -> int:
        """Count the queries that get an empty list because of no frame."""
        return sum(1 for query in queries if not self.upstream_repos(query))


def rank_queries(
    rule: FrameRule,
    queries: Iterable[Query],
    exclude_same_repo: bool = True,
    top_k: int = 100,
) -> dict[str, list[str]]:
    """Return the run: query ID to ranked candidate IDs."""
    return {
        query.id: rule.search(query, exclude_same_repo, top_k) for query in queries
    }