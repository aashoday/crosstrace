import tomllib
from dataclasses import dataclass
from pathlib import Path

from crosstrace.config import ROOT

VALID_ROLES = {"upstream", "downstream", "both"}


@dataclass(frozen=True)
class Project:
    repo: str
    role: str

    @property
    def owner(self) -> str:
        return self.repo.split("/")[0]

    @property
    def name(self) -> str:
        return self.repo.split("/")[1]


def load_projects(path: Path = ROOT / "projects.toml") -> list[Project]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    projects = []
    seen = set()
    for item in raw["project"]:
        repo, role = item["repo"], item["role"]
        if role not in VALID_ROLES:
            raise ValueError(f"Bad role for {repo}: {role}")
        if repo.count("/") != 1:
            raise ValueError(f"Repo must look like owner/name: {repo}")
        if repo in seen:
            raise ValueError(f"Duplicate repo: {repo}")
        seen.add(repo)
        projects.append(Project(repo, role))
    return projects