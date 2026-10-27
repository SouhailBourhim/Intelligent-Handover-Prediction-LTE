"""
Assemble docs/cours/*.md into one HTML page and print it to PDF.

Usage
-----
    python scripts/build_course_pdf.py

Needs `markdown` and `pymdown-extensions` (requirements-dev.txt) and a local
Microsoft Edge or Google Chrome, used headless to render the Mermaid diagrams
and MathJax formulas (loaded from a CDN) and to print the page.

Output: docs/cours/cours_projet_handover_lte.pdf
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT    = Path(__file__).resolve().parents[1]
COURSE  = ROOT / "docs" / "cours"
BUILD   = COURSE / "_build"
PDF     = COURSE / "cours_projet_handover_lte.pdf"

CHAPTERS = sorted(p for p in COURSE.glob("[0-9][0-9]_*.md"))

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.5; color: #1f2933; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 20pt; border-bottom: 3px solid #3b82f6; padding-bottom: 4px;
     page-break-before: always; margin-top: 0; }
h1.cover { page-break-before: avoid; border: none; font-size: 26pt; margin-top: 60mm; }
h2 { font-size: 14pt; color: #1e40af; margin-top: 18px; }
h3 { font-size: 12pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #cbd2d9; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef2f7; }
img { max-width: 100%; display: block; margin: 8px auto; page-break-inside: avoid; }
blockquote { border-left: 4px solid #3b82f6; background: #f5f8ff; margin: 8px 0;
             padding: 6px 12px; page-break-inside: avoid; }
code { background: #f1f5f9; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f8fafc; border: 1px solid #e4e7eb; padding: 8px; font-size: 8.5pt;
      white-space: pre-wrap; page-break-inside: avoid; }
pre.mermaid { background: white; border: none; text-align: center; }
/* a diagram taller than the page would be pushed to the next page and dropped */
pre.mermaid svg { max-height: 230mm; max-width: 100% !important; width: auto; height: auto; }
hr { border: none; border-top: 1px solid #e4e7eb; }
.cover-sub { font-size: 13pt; color: #52606d; }
"""

HEAD = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Cours — Prédiction intelligente du handover LTE</title>
<style>{css}</style>
<script>window.MathJax = {{ tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }},
  svg: {{ fontCache: 'global' }} }};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
</head><body>
<h1 class="cover">Prédiction intelligente du handover dans les réseaux LTE</h1>
<p class="cover-sub">Cours complet — télécommunications mobiles et apprentissage automatique,
construit à partir du projet <em>Intelligent-Handover-Prediction-LTE</em>.</p>
"""

# Classic (blocking) script: headless printing waits for it, whereas an ES
# module loads asynchronously and some diagrams were printed blank.
TAIL = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({ startOnLoad: true, theme: "neutral", flowchart: { useMaxWidth: true },
                     sequence: { useMaxWidth: true } });
</script></body></html>
"""


def convert(md_text: str) -> str:
    # Mermaid blocks become <pre class="mermaid"> before markdown sees them
    blocks: list[str] = []

    def keep(m):
        blocks.append(m.group(1))
        return f"\n\nMERMAIDBLOCK{len(blocks) - 1}\n\n"

    md_text = re.sub(r"```mermaid\n(.*?)```", keep, md_text, flags=re.S)
    html = markdown.markdown(md_text, extensions=[
        "tables", "fenced_code", "attr_list", "sane_lists",
        "pymdownx.arithmatex",          # protects $…$ / $$…$$ for MathJax
    ], extension_configs={"pymdownx.arithmatex": {"generic": True}})
    for i, b in enumerate(blocks):
        safe = b.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = html.replace(f"<p>MERMAIDBLOCK{i}</p>", f'<pre class="mermaid">{safe}</pre>')
    # navigation links between chapters are noise in a printed document
    html = re.sub(r"<p>(⬅️|➡️).*?</p>", "", html, flags=re.S)
    return html


def build_html() -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    parts = [HEAD.format(css=CSS)]
    for ch in CHAPTERS:
        parts.append(convert(ch.read_text(encoding="utf-8")))
    parts.append(TAIL)
    out = COURSE / "_cours.html"          # next to the chapters so img/ paths resolve
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def print_pdf(html: Path):
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("chrome")
    if not browser:
        sys.exit("No Edge/Chrome found — open docs/cours/_cours.html and print it to PDF.")
    subprocess.run([
        browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--virtual-time-budget=45000", "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF}", html.resolve().as_uri(),
    ], check=True, timeout=300)


if __name__ == "__main__":
    print(f"Chapters: {[c.name for c in CHAPTERS]}")
    html = build_html()
    print(f"HTML  -> {html.relative_to(ROOT)}")
    print_pdf(html)
    print(f"PDF   -> {PDF.relative_to(ROOT)}  ({PDF.stat().st_size / 1e6:.1f} MB)")
