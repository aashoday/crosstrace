import csv
import random
import sqlite3
from collections import Counter
from pathlib import Path

import httpx

from crosstrace.config import ROOT
from crosstrace.github import GitHubClient

LABELS_PATH = ROOT / "labels" / "audit.csv"
FIELDS = [
    "source_repo",
    "source_number",
    "target_repo",
    "target_number",
    "label",
    "note",
]
NAMES = {"c": "cause", "r": "related", "n": "noise", "x": "unclear"}

Pair = tuple[str, int, str, int]


def sample_pairs(conn: sqlite3.Connection, repos: set[str], seed: int = 42) -> list[Pair]:
    rows = conn.execute(
        "SELECT source_repo, source_number, target_repo, target_number "
        "FROM cross_refs "
        "ORDER BY source_repo, source_number, target_repo, target_number"
    ).fetchall()
    pairs = [row for row in rows if row[0] in repos and row[2] in repos]
    random.Random(seed).shuffle(pairs)
    return pairs


def load_labels(path: Path = LABELS_PATH) -> dict[Pair, str]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {
            (
                row["source_repo"],
                int(row["source_number"]),
                row["target_repo"],
                int(row["target_number"]),
            ): row["label"]
            for row in csv.DictReader(f)
        }


def save_label(pair: Pair, label: str, note: str = "", path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(FIELDS)
        writer.writerow([*pair, label, note])


def next_unlabeled(pairs: list[Pair], labels: dict[Pair, str]) -> Pair | None:
    for pair in pairs:
        if pair not in labels:
            return pair
    return None


def summarize(labels: dict[Pair, str]) -> str:
    counts = Counter(labels.values())
    total = sum(counts.values())
    lines = [f"Labeled: {total}"]
    for name in NAMES.values():
        n = counts.get(name, 0)
        share = 100 * n / total if total else 0
        lines.append(f"  {name:8s} {n:4d}  {share:5.1f}%")
    return "\n".join(lines)


def fetch_issue(client: GitHubClient, repo: str, number: int) -> dict | None:
    try:
        return client.get(f"/repos/{repo}/issues/{number}")["data"]
    except httpx.HTTPStatusError:
        return None


def excerpt(text: str | None, size: int = 600) -> str:
    text = (text or "").strip().replace("\r", "")
    return text if len(text) <= size else text[:size] + " ..."


def context(text: str | None, repo: str, number: int, width: int = 200) -> str | None:
    if not text:
        return None
    needles = [f"{repo}#{number}", f"{repo}/issues/{number}", f"{repo}/pull/{number}"]
    for needle in needles:
        index = text.find(needle)
        if index != -1:
            start = max(0, index - width)
            return text[start : index + len(needle) + width].replace("\n", " ")
    return None