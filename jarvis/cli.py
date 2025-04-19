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
import click
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
        "Ask questions about python codebase using `/scancode <path>`"
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
@click.command()
@click.option(
    "--model", default="gemma3:4b", help="Ollama model name to use", show_default=True
)
@click.option(
    "--context",
    "context_window",
    default=8192,
    type=int,
    help="Maximum context window size",
    show_default=True,
)
def cli(model: str, context_window: int) -> None:
    """
    Jarvis CLI for chatting with a local Ollama model.
    """
    console.print(
        f"[bold cyan]Jarvis Local CLI[/] — model: [magenta]{model}[/], context: [yellow]{context_window} tokens[/]\n"
        "Press Ctrl‑D to quit, Ctrl‑C to cancel\n"
    )

    from .client import JarvisClient
    client = JarvisClient(model=model, context_size = context_window)

    while True:
        question = _read_multiline_question()
        if not question.strip():
            continue

        if question.strip().lower().startswith("/scancode"):
            parts = question.split(maxsplit=1)
            client.scan_codebase(parts[1])
            continue

        if question.strip().lower().endswith("/new"):
            client.reset()
            console.print("[cyan]🔄  New chat started.[/]\n")
            continue

        _render_response(client.ask(question))


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\nGood‑bye 👋", style="cyan")
        sys.exit(0)