from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schema import relative_path


TreeItem = tuple[str, Sequence["TreeItem"]]


def kv(key: str, value: Any) -> TreeItem:
    return (f"{key}={format_value(value)}", [])


def section(name: str, status: str, attrs: Sequence[TreeItem] | None = None) -> TreeItem:
    return (f"{name} {status}", list(attrs or []))


def format_tree(title: str, items: Sequence[TreeItem], level: str = "INFO") -> str:
    lines = [f"[{level}] {title}"]
    _append_items(lines, list(items), prefix="")
    return "\n".join(lines)


def log_tree(title: str, items: Sequence[TreeItem], level: str = "INFO") -> None:
    print(format_tree(title, items, level=level))


def write_tree_log(path: Path, title: str, items: Sequence[TreeItem], level: str = "INFO") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_tree(title, items, level=level) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, Path):
        return relative_path(value, Path.cwd())
    return str(value).replace("\n", "\\n")


def _append_items(lines: list[str], items: list[TreeItem], prefix: str) -> None:
    for idx, (label, children) in enumerate(items):
        is_last = idx == len(items) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{label}")
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        _append_items(lines, list(children), child_prefix)
