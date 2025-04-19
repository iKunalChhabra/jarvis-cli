"""
jarvis.code.scanner
===================

Walks the working directory, honours ``.gitignore``, collects every
``*.py`` file, and produces a Markdown snapshot suitable for LLM context.

Time / Space complexity
-----------------------
* Scanning  : **O(F + S)** F = number of Python files, S = total size (bytes).
* Snapshot  : **O(Snap)** Snap = characters retained (capped below).
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable, List, Sequence

import pathspec  # pip install pathspec

__all__ = ["CodeScanner", "CodeSnapshot"]

#: Hard ceiling per file – avoids single‑file context explosions
MAX_CHARS_PER_FILE: int = 10_000
#: Global ceiling – protects overall prompt budget
MAX_TOTAL_CHARS: int = 60_000


# ───────────────────────────────── helpers ───────────────────────────────── #

class GitIgnoreFilter:
    """Returns *True* for paths that must be skipped according to .gitignore."""

    def __init__(self, root: Path) -> None:
        gi = root / ".gitignore"
        patterns: Sequence[str] = gi.read_text().splitlines() if gi.exists() else []
        self._spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def match(self, rel_path: Path) -> bool:
        """Check whether *rel_path* (POSIX style) is ignored."""
        return self._spec.match_file(rel_path.as_posix())


class CodeSnapshot(str):
    """
    Immutable Markdown document representing a *slim* view of the repo:
    * first a tree‑view list,
    * then each file embedded in ```python fences.
    """

    @staticmethod
    def from_paths(root: Path, files: Iterable[Path]) -> "CodeSnapshot":
        tree: List[str] = ["# Project layout\n"]
        for f in sorted(files):
            tree.append(f"* {f.as_posix()}")
        md = "\n".join(tree) + "\n\n# Files\n"

        total = len(md)
        body: List[str] = [md]

        for f in files:
            code = f.read_text(encoding="utf‑8", errors="replace")
            if len(code) > MAX_CHARS_PER_FILE:
                code = textwrap.shorten(code, MAX_CHARS_PER_FILE, placeholder="\n# …truncated…\n")

            snippet = f"```python {f.as_posix()}\n{code}\n```\n\n"
            if total + len(snippet) > MAX_TOTAL_CHARS:
                body.append("*Further files omitted to fit context*\n")
                break

            body.append(snippet)
            total += len(snippet)

        return CodeSnapshot("".join(body))


class CodeScanner:
    """Facade that produces a `CodeSnapshot` for the directory tree."""

    def __init__(self, root: str | Path = ".") -> None:
        self._root = Path(root).resolve()
        self._filter = GitIgnoreFilter(self._root)

    # public API ------------------------------------------------------------ #
    def scan(self) -> CodeSnapshot:
        print("Analysing codebase at : ", self._root)
        py_files: List[Path] = []
        for path in self._root.rglob("*.py"):
            print("Reading file : ", path)
            rel = path.relative_to(self._root)
            if not self._filter.match(rel):
                py_files.append(rel)
        return CodeSnapshot.from_paths(self._root, py_files)