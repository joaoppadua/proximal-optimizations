"""Small helpers the notebooks share."""

import inspect
import textwrap
from collections.abc import Callable

import marimo as mo


def source_md(fn: Callable) -> str:
    """The function's source in a fenced Python block, ready for `mo.md`."""
    source = textwrap.dedent(inspect.getsource(fn)).rstrip()
    return f"```python\n{source}\n```"


def reveal(question: str, answer: str) -> mo.Html:
    """The question as text, with the answer folded inside a collapsed accordion."""
    return mo.vstack([mo.md(question), mo.accordion({"Show the answer": mo.md(answer)})])
