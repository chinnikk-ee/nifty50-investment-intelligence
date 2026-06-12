"""Render docs/TECHNICAL_REPORT.md to a paginated PDF (Deliverable 2).

Dependency-free: uses only matplotlib (already a project dependency), the same
backend as ml/reports.py. Supports a pragmatic subset of Markdown sufficient for
the technical report: H1/H2/H3 headings, paragraphs, blockquotes, bullet lists,
pipe tables and image embeds. Content flows across A4 pages automatically.

Usage:  python scripts/build_report.py
Output: reports/generated/TECHNICAL_REPORT.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "TECHNICAL_REPORT.md"
OUT = ROOT / "reports" / "generated" / "TECHNICAL_REPORT.pdf"

# A4 portrait in inches; layout in figure-fraction coordinates.
PAGE = (8.27, 11.69)
LEFT, RIGHT, TOP, BOTTOM = 0.08, 0.92, 0.94, 0.07
WIDTH = RIGHT - LEFT
LINE = 0.020          # vertical step for one body line (figure fraction)
CHARS_PER_LINE = 95   # wrap width for body text at 10pt


def parse_blocks(md: str):
    """Turn markdown into a flat list of (kind, payload) blocks."""
    blocks, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s == "---":
            blocks.append(("rule", None)); i += 1; continue
        m = re.match(r"^(#{1,3})\s+(.*)", s)
        if m:
            blocks.append((f"h{len(m.group(1))}", m.group(2))); i += 1; continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", s)
        if m:
            blocks.append(("img", (m.group(1), m.group(2)))); i += 1; continue
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            blocks.append(("quote", " ".join(buf))); continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip()); i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c or "-") for c in r)]
            blocks.append(("table", cells)); continue
        if re.match(r"^[-*]\s+", s):
            buf = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                buf.append(re.sub(r"^[-*]\s+", "", lines[i].strip())); i += 1
            blocks.append(("ul", buf)); continue
        # paragraph: gather until blank
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,3}\s|[-*]\s|\||>|!\[|---$)", lines[i].strip()
        ):
            buf.append(lines[i].strip()); i += 1
        blocks.append(("p", " ".join(buf)))
    return blocks


def clean(text: str) -> str:
    """Strip inline markdown that we render as plain styled text."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text


def wrap(text: str, width: int):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


class Renderer:
    def __init__(self, pdf):
        self.pdf = pdf
        self._new_page()

    def _new_page(self):
        self.fig = plt.figure(figsize=PAGE)
        self.y = TOP

    def _flush(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def _ensure(self, need):
        if self.y - need < BOTTOM:
            self._flush(); self._new_page()

    def text(self, s, *, size=10, weight="normal", color="black", dy=LINE, x=LEFT, style="normal"):
        self._ensure(dy)
        self.fig.text(x, self.y, s, fontsize=size, fontweight=weight, color=color,
                      style=style, ha="left", va="top", wrap=False)
        self.y -= dy

    def gap(self, h=LINE * 0.6):
        self.y -= h

    def render(self, blocks):
        for kind, payload in blocks:
            if kind == "h1":
                for w in wrap(clean(payload), 42):
                    self.text(w, size=19, weight="bold", dy=LINE * 1.5)
                self.gap(LINE * 0.4)
            elif kind == "h2":
                self.gap(LINE * 0.8)
                self._ensure(LINE * 2.5)
                self.text(clean(payload), size=14, weight="bold", color="#1e293b", dy=LINE * 1.4)
                self._rule(thin=True)
            elif kind == "h3":
                self.gap(LINE * 0.4)
                self.text(clean(payload), size=11.5, weight="bold", color="#334155", dy=LINE * 1.2)
            elif kind == "p":
                for w in wrap(clean(payload), CHARS_PER_LINE):
                    self.text(w)
                self.gap()
            elif kind == "quote":
                for w in wrap(clean(payload), CHARS_PER_LINE - 6):
                    self.text(w, size=9, color="#64748b", style="italic", x=LEFT + 0.02)
                self.gap()
            elif kind == "ul":
                for item in payload:
                    lines = wrap(clean(item), CHARS_PER_LINE - 4)
                    self.text("• " + lines[0], x=LEFT + 0.01)
                    for extra in lines[1:]:
                        self.text("  " + extra, x=LEFT + 0.01)
                self.gap()
            elif kind == "table":
                self._table(payload)
            elif kind == "img":
                self._image(payload[1], payload[0])
            elif kind == "rule":
                pass  # section breaks handled by h2 underlines

    def _rule(self, thin=False):
        self._ensure(LINE)
        self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [self.y + LINE * 0.4] * 2,
                                       color="#cbd5e1", lw=0.8 if thin else 1.4,
                                       transform=self.fig.transFigure))
        self.y -= LINE * 0.5

    def _table(self, rows):
        if not rows:
            return
        ncol = max(len(r) for r in rows)
        rows = [[clean(c) for c in r] + [""] * (ncol - len(r)) for r in rows]
        # proportional column widths from max content length, clamped
        maxlen = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(ncol)]
        total = sum(maxlen) or 1
        colw = [max(0.10, min(0.45, m / total)) for m in maxlen]
        scale = WIDTH / sum(colw)
        colw = [w * scale for w in colw]
        xstart = [LEFT + sum(colw[:c]) for c in range(ncol)]
        chars = [int(colw[c] / 0.0062) for c in range(ncol)]  # ~chars per col at 8pt

        line_h = LINE * 1.05
        for ri, row in enumerate(rows):
            wrapped = [wrap(row[c], max(6, chars[c])) for c in range(ncol)]
            rh = line_h * max(len(w) for w in wrapped) + line_h * 0.5
            self._ensure(rh + (line_h if ri == 0 else 0))
            yy = self.y
            if ri == 0:
                self.fig.patches.append(plt.Rectangle((LEFT, yy - rh), WIDTH, rh,
                                        transform=self.fig.transFigure,
                                        facecolor="#1e293b", edgecolor="none"))
            for ci in range(ncol):
                for li, seg in enumerate(wrapped[ci]):
                    self.fig.text(xstart[ci] + 0.006, yy - line_h * (0.7 + li), seg,
                                  fontsize=8, va="center", ha="left",
                                  color="white" if ri == 0 else "#111827",
                                  fontweight="bold" if ri == 0 else "normal")
            self.fig.add_artist(plt.Line2D([LEFT, RIGHT], [yy - rh] * 2, color="#e2e8f0",
                                lw=0.5, transform=self.fig.transFigure))
            self.y = yy - rh
        self.y -= LINE * 0.6

    def _image(self, rel, caption):
        path = (SRC.parent / rel).resolve()
        if not path.exists():
            self.text(f"[missing image: {rel}]", size=8, color="red"); return
        img = mpimg.imread(path)
        ih, iw = img.shape[0], img.shape[1]
        disp_w = WIDTH
        disp_h = disp_w * (ih / iw) * (PAGE[0] / PAGE[1])
        disp_h = min(disp_h, 0.30)  # cap so a figure never dominates a page
        self._ensure(disp_h + LINE * 1.5)
        ax = self.fig.add_axes([LEFT, self.y - disp_h, disp_w, disp_h])
        ax.imshow(img); ax.axis("off")
        self.y -= disp_h + LINE * 0.2
        if caption:
            self.text("Figure: " + clean(caption), size=8, color="#64748b", style="italic")
        self.gap()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    md = SRC.read_text(encoding="utf-8")
    blocks = parse_blocks(md)
    with PdfPages(OUT) as pdf:
        r = Renderer(pdf)
        r.render(blocks)
        r._flush()
    # report page count
    try:
        from pypdf import PdfReader
        n = len(PdfReader(str(OUT)).pages)
    except Exception:
        n = "?"
    print(f"wrote {OUT}  ({n} pages)")


if __name__ == "__main__":
    main()
