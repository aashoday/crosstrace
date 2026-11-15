# Contributing to CrossTrace

## Workflow

1. Do not push to `main`. All changes go through a pull request (PR).
2. Make one branch for each change: `type/short-description`.
3. Keep each PR small. One PR does one thing.
4. Merge with squash merge. Delete the branch after the merge.

## PR title tags

Start every PR title with one tag:

- `[ENH]` new feature
- `[BUG]` bug fix
- `[MNT]` maintenance, tooling, CI
- `[DOC]` documentation
- `[REF]` refactor with no behavior change

Example: `[MNT] Add contribution workflow`

## Local setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Undo a change

Each merged PR is one commit on `main`. To undo it, use the **Revert** button on the PR page. Revert makes a new PR.