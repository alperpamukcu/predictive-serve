from __future__ import annotations

import base64
import html
import re


def initials_from_display_name(name: str, max_chars: int = 2) -> str:
    s = str(name).replace("\u00a0", " ").strip()
    if not s:
        return "?"
    chunks = [c.strip() for c in re.split(r"[,/]|(?:\s+v(?:s|S)\s+)", s) if c.strip()]
    picks: list[str] = []
    for ch in chunks[:2]:
        token = ch.split()[0]
        picks.append(token[0].upper() if token else "")
    out = "".join([p for p in picks if p])
    out = out or "".join(ch for ch in s[:3].upper() if ch.isalnum())
    return (out[: max(1, int(max_chars))] or "?")


def svg_avatar_data_uri(display_name: str, size: int = 128, gradient_a: str = "#1e3a5f", gradient_b: str = "#243b76") -> str:
    initials = initials_from_display_name(display_name, max_chars=2)
    initials_esc = html.escape(initials)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<defs>'
        f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{gradient_a}"/>'
        f'<stop offset="1" stop-color="{gradient_b}"/>'
        f"</linearGradient>"
        f"</defs>"
        f'<rect width="{size}" height="{size}" rx="{size//2}" fill="url(#bg)"/>'
        f'<text x="50%" y="52%" dominant-baseline="middle" text-anchor="middle" '
        f'fill="#e8eaf0" font-family="system-ui, -apple-system, Segoe UI, sans-serif" '
        f'font-size="{int(size*0.36)}" font-weight="700">{initials_esc}</text>'
        f"</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"
