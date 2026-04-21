from __future__ import annotations

import html
import re


INLINE_CODE_RE = re.compile(r"`([^`]+)`")
STRONG_RE = re.compile(r"\*\*([^*]+)\*\*")
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _format_inline(text: str) -> str:
    parts = INLINE_CODE_RE.split(text)
    formatted_parts: list[str] = []

    for index, part in enumerate(parts):
        if index % 2 == 1:
            formatted_parts.append(f"<code>{html.escape(part)}</code>")
            continue

        escaped = html.escape(part)
        escaped = STRONG_RE.sub(lambda m: f"<strong>{html.escape(m.group(1))}</strong>", escaped)
        escaped = LINK_RE.sub(
            lambda m: f'<a href="{html.escape(m.group(2), quote=True)}" target="_blank" rel="noopener">{html.escape(m.group(1))}</a>',
            escaped,
        )
        formatted_parts.append(escaped)

    return "".join(formatted_parts)


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    blocks: list[str] = []
    list_mode: str | None = None

    def close_list() -> None:
        nonlocal list_mode
        if list_mode is not None:
            blocks.append(f"</{list_mode}>")
            list_mode = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            close_list()
            continue

        if stripped.startswith("### "):
            close_list()
            blocks.append(f"<h3>{_format_inline(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            close_list()
            blocks.append(f"<h2>{_format_inline(stripped[3:])}</h2>")
            continue
        if stripped.startswith("# "):
            close_list()
            blocks.append(f"<h1>{_format_inline(stripped[2:])}</h1>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            if list_mode != "ol":
                close_list()
                list_mode = "ol"
                blocks.append("<ol>")
            item_text = re.sub(r"^\d+\.\s+", "", stripped)
            blocks.append(f"<li>{_format_inline(item_text)}</li>")
            continue

        if stripped.startswith("- "):
            if list_mode != "ul":
                close_list()
                list_mode = "ul"
                blocks.append("<ul>")
            blocks.append(f"<li>{_format_inline(stripped[2:])}</li>")
            continue

        close_list()
        blocks.append(f"<p>{_format_inline(stripped)}</p>")

    close_list()
    return "\n".join(blocks)
