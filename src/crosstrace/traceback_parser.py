"""Parse Python tracebacks and package versions from issue text."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ALIASES = {"sklearn": "scikit-learn", "skimage": "scikit-image"}
_SEPARATORS = re.compile(r"[-_.]+")
_SITE_DIRS = frozenset({"site-packages", "dist-packages"})

_HEADER = re.compile(r"^\s*Traceback \(most recent call last\):\s*$")
_IPY_HEADER = re.compile(
    r"^\s*[A-Za-z_][\w.]*\s+Traceback \(most recent call last\)\s*$"
)
_FILE = re.compile(
    r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+?))?\s*$'
)
_IPY_FILE = re.compile(
    r"^\s*File (?P<path>\S.*?):(?P<line>\d+)(?:, in (?P<func>.+?))?\s*$"
)
_ERROR = re.compile(
    r"^(?P<type>[A-Za-z_][\w.]*?"
    r"(?:Error|Exception|Warning|Exit|Interrupt|Iteration))"
    r"(?::\s*(?P<message>.*))?$"
)
_VERSION_LINE = re.compile(
    r"^\s*(?:[-*]\s+)?(?P<name>[A-Za-z][\w.\-]*)"
    r"(?:\s*==\s*|\s*:\s*|\s+)"
    r"(?P<version>\d+\.\d[\w.+!\-]*)\s*$"
)


@dataclass(frozen=True)
class Frame:
    """One frame of a traceback."""

    path: str
    line: int
    function: str | None
    package: str | None


@dataclass(frozen=True)
class Traceback:
    """One traceback. Frames are in the printed order (oldest call first)."""

    frames: tuple[Frame, ...]
    error_type: str | None
    error_message: str | None

    @property
    def packages(self) -> tuple[str, ...]:
        """Package names in frame order. Each name appears once."""
        names: list[str] = []
        for frame in self.frames:
            if frame.package and frame.package not in names:
                names.append(frame.package)
        return tuple(names)


@dataclass(frozen=True)
class ParsedIssue:
    """All results for one issue text."""

    tracebacks: tuple[Traceback, ...]
    versions: dict[str, str]


def canonical_name(name: str) -> str:
    """Make a package name lower case, with '-' as the only separator."""
    cleaned = _SEPARATORS.sub("-", name.strip()).lower()
    return _ALIASES.get(cleaned, cleaned)


def package_from_path(path: str) -> str | None:
    """Return the package name for an installed file, or None."""
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts):
        if part not in _SITE_DIRS:
            continue
        rest = parts[index + 1 :]
        while rest and rest[0].endswith(".egg"):
            rest = rest[1:]
        if not rest:
            return None
        name = rest[0]
        if len(rest) == 1 and name.endswith(".py"):
            name = name[:-3]
        return canonical_name(name)
    return None


def _match_frame(line: str, ipython: bool) -> Frame | None:
    match = _FILE.match(line)
    if match:
        function = match["func"]
    elif ipython:
        match = _IPY_FILE.match(line)
        if match is None:
            return None
        function = (match["func"] or "").split("(", 1)[0].strip()
    else:
        return None
    path = match["path"]
    return Frame(
        path=path,
        line=int(match["line"]),
        function=function or None,
        package=package_from_path(path),
    )


def _read_block(
    lines: list[str], start: int, ipython: bool
) -> tuple[Traceback, int]:
    """Read one traceback. Return it and the index of the next line to read."""
    frames: list[Frame] = []
    error_type: str | None = None
    error_message: str | None = None
    index = start
    while index < len(lines):
        line = lines[index]
        if _HEADER.match(line) or _IPY_HEADER.match(line):
            break
        frame = _match_frame(line, ipython)
        if frame:
            frames.append(frame)
            index += 1
            continue
        if not line.strip():
            if not ipython:
                break
            index += 1
            continue
        if line[0].isspace():
            index += 1
            continue
        match = _ERROR.match(line)
        if match:
            error_type = match["type"]
            error_message = match["message"] or None
            index += 1
            break
        if not ipython:
            break
        index += 1
    return Traceback(tuple(frames), error_type, error_message), index


def parse_tracebacks(text: str) -> list[Traceback]:
    """Find all tracebacks in the text. Return [] if there are none."""
    lines = text.splitlines()
    result: list[Traceback] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _HEADER.match(line):
            ipython = False
        elif _IPY_HEADER.match(line):
            ipython = True
        else:
            index += 1
            continue
        traceback, index = _read_block(lines, index + 1, ipython)
        result.append(traceback)
    return result


def parse_versions(text: str) -> dict[str, str]:
    """Find 'name version' pairs. The first value for a name wins."""
    versions: dict[str, str] = {}
    for line in text.splitlines():
        match = _VERSION_LINE.match(line)
        if match:
            versions.setdefault(canonical_name(match["name"]), match["version"])
    return versions


def parse_issue_text(text: str) -> ParsedIssue:
    """Parse tracebacks and versions from one issue text."""
    return ParsedIssue(
        tracebacks=tuple(parse_tracebacks(text)),
        versions=parse_versions(text),
    )