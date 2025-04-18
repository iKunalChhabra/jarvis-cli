"""
jarvis/cli.py
=============

Command‑line interface for **Jarvis Φ‑3.5** running on a local Ollama server.
Features
--------
* Multi‑line prompt entry (paste blocks, hit *blank line* or `/send` to submit)
* `/new` command resets conversation history
* Coloured inline prefix “🤖 Jarvis ‑”
* Streaming answers with Markdown rendering
* Ctrl‑C while **typing**  → draft cleared (stay in prompt)
* Ctrl‑C during **answer** → cancel completion (keeps CLI alive)
* Ctrl‑D (or Ctrl‑Z+Enter on Windows) → quit program
"""

from __future__ import annotations

import sys
from contextlib import suppress
from typing import Iterator, List

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

# ---------------------------------------------------------------------------

console = Console()
PREFIX_TEXT = Text("🤖 Jarvis - ", style="bold cyan")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _read_multiline_question() -> str:
    """
    Read a prompt of arbitrary length from ``stdin`` **synchronously**.

    Terminators
    -----------
    * Blank line **after** ≥ 1 line of text, or
    * A line containing only ``/send`` or ``/new`` (case‑insensitive).

    Special keys
    ------------
    * Ctrl‑C → abandon current draft, return ``""`` (no request sent)
    * Ctrl‑D → exit program immediately

    Complexity
    ----------
    Time   O(N)  where *N* = input characters
    Memory O(N)
    """
    console.print(
        "[bold magenta]You[/] "
        "(multi‑line allowed; end with blank line, /send or /new; Ctrl‑D to quit)"
    )

    lines: List[str] = []
    while True:
        try:
            line = input()
        except KeyboardInterrupt:           # Ctrl‑C while typing
            console.print("[red]⏹️  Draft cleared (Ctrl‑C)[/]\n")
            return ""
        except EOFError:                    # Ctrl‑D / Ctrl‑Z+Enter
            console.print("\nGood‑bye 👋", style="cyan")
            raise SystemExit(0)

        sentinel = line.strip().lower()
        if sentinel in {"/send", "/new"} or (line == "" and lines):
            lines.append(line)              # keep sentinel/blank for caller
            break
        if line == "" and not lines:        # stray blank at start
            continue

        lines.append(line)

    return "\n".join(lines).rstrip("\n")


# ──────────────────────────────────────────────────────────────────────────
# Streaming renderer
# ──────────────────────────────────────────────────────────────────────────
def _render_response(stream: Iterator[str]) -> None:
    """
    Stream tokens from *stream* while showing a **persistent** coloured prefix.

    Time   O(L)  L = number of tokens received
    Memory O(A)  A = size of final answer
    """
    answer: List[str] = []
    with Live("", console=console, refresh_per_second=8) as live:
        try:
            for chunk in stream:
                answer.append(chunk)
                live.update(Group(PREFIX_TEXT, Markdown("".join(answer))))
        except KeyboardInterrupt:           # Ctrl‑C cancels completion
            live.stop()
            console.print("\n[red]⏹️  Completion cancelled (Ctrl‑C)[/]")
            with suppress(Exception):
                for _ in stream:            # drain generator to close socket
                    pass

    console.print()                         # tidy newline after answer


# ──────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:  # pragma: no cover
    console.print(
        "[bold cyan]Jarvis Local CLI[/] — press Ctrl‑D to quit, Ctrl‑C to cancel\n"
    )

    # Lazy import to avoid circular references during unit tests
    from .client import JarvisClient

    client = JarvisClient()

    while True:
        question = _read_multiline_question()

        # Detect sentinel commands (may be last line of buffer)
        if question.strip().lower().endswith("/new"):
            client.reset()
            console.print("[cyan]🔄  New chat started.[/]\n")
            continue

        if not question.strip():            # empty or cancelled draft
            continue

        _render_response(client.ask(question))


# ──────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":                  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nGood‑bye 👋", style="cyan")
        sys.exit(0)