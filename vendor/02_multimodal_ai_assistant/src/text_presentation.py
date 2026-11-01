"""Conservative Markdown-to-text presentation shared by chat and speech.

This is a stateless presentation helper, not an HTML or Markdown renderer.
Call it with the accumulated model response, rather than individual tokens.
"""
from __future__ import annotations

import re


_LINK = re.compile(r"!?\[([^\]\n]*)\]\((?:[^()\n]|\([^()\n]*\))*\)")
_FENCE = re.compile(r"^\s*(?:`{3,}|~{3,})[^\n]*$")


def _table_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return len(cells) > 1 and all(
        re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells
    )


def _strip_once(text: str) -> str:
    lines = text.split("\n")
    table_rows: set[int] = set()
    for index, line in enumerate(lines):
        if _table_separator(line):
            # A separator identifies borderless tables without changing x | y.
            for direction in (-1, 1):
                row = index + direction
                while 0 <= row < len(lines) and "|" in lines[row]:
                    table_rows.add(row)
                    row += direction

    result = []
    for index, line in enumerate(lines):
        if _FENCE.fullmatch(line) or _table_separator(line):
            continue
        if index in table_rows or (
            line.strip().startswith("|") and line.strip().endswith("|")
            and line.strip().count("|") >= 2
        ):
            line = "：".join(cell.strip() for cell in line.strip().strip("|").split("|"))

        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+)$", line)
        if heading:
            line = re.sub(r"\s+#+\s*$", "", heading[1])
        line = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", line)
        line = _LINK.sub(r"\1", line)
        line = re.sub(r"<(https?://[^\s<>]+)>", r"\1", line)
        line = line.replace("`", "")
        # Non-space edges distinguish emphasis from 2 ** 3 and a * b.
        line = re.sub(r"(?<![A-Za-z0-9_*])\*\*(\S(?:.*?\S)?)\*\*(?![A-Za-z0-9_*])", r"\1", line)
        # ASCII dunder identifiers are common code even outside backticks.
        line = re.sub(
            r"(?<!\w)__(\S(?:.*?\S)?)__(?!\w)",
            lambda match: match[0] if re.fullmatch(r"[A-Za-z_]\w*", match[1], re.ASCII) else match[1],
            line,
        )
        line = re.sub(r"(?<![A-Za-z0-9_*])\*(\S(?:.*?\S)?)\*(?![A-Za-z0-9_*])", r"\1", line)
        line = re.sub(r"(?<!\w)_([^_\s](?:.*?[^_\s])?)_(?!\w)", r"\1", line)
        # An opening bold marker at a streaming boundary need not leak to UI.
        line = re.sub(r"(?<![A-Za-z0-9_*])\*\*(?=\S)", "", line)
        result.append(line)
    return "\n".join(result)


def to_plain_text(text: str) -> str:
    """Return readable text without common model-generated Markdown syntax.

    Preserve literal HTML, entities, comparison operators, ordinary underscores
    and arithmetic. No HTML decoding occurs. Repeated calls are idempotent;
    incomplete opening bold markers and code fences are tolerated while the
    caller continues accumulating the original stream.
    """
    previous = text.replace("\r\n", "\n").replace("\r", "\n")
    while True:
        plain = _strip_once(previous)
        if plain == previous:
            return plain
        previous = plain


def speech_ready_text(text: str, *, final: bool = False) -> str:
    """Project only text whose Markdown interpretation can no longer change.

    A table header is ambiguous until its block closes. A partial link can
    contain sentence punctuation in its URL. Keep these tails out of the
    sentence queue until they can be converted as a whole.
    """
    if final:
        return to_plain_text(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    limit = len(text)
    lines = text.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        if '|' in line:
            closed = any('|' not in following and following.endswith('\n')
                         for following in lines[index + 1:])
            if not closed:
                limit = min(limit, offset)
                break
        offset += len(line)
    complete_links = list(_LINK.finditer(text))
    for match in re.finditer(r'\[', text):
        if any(link.start() <= match.start() < link.end() for link in complete_links):
            continue
        if '\n' not in text[match.start():]:
            limit = min(limit, match.start())
    # Inline emphasis is line-local; its final closing marker may shorten the
    # projection. Plain sentences still stream immediately without a newline.
    line_start = text.rfind('\n') + 1
    if '*' in text[line_start:] or '_' in text[line_start:]:
        limit = min(limit, line_start)
    return to_plain_text(text[:limit])
