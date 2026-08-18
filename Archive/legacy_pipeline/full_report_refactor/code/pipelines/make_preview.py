"""Build a readable PDF from overleaf/main.tex without the publisher class.

The report targets `opticajnl.cls`, which ships with the Optica/Overleaf template
and is not in TeX Live, so it cannot be compiled locally. Rather than keep a
second copy of the manuscript that would drift, this script rewrites the
preamble of the real source on the fly and compiles that. The body, tables,
figures, and bibliography are the same bytes as the submission file.

Usage:  python3 pipelines/make_preview.py
Output: final_full_report_preview.pdf
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "overleaf" / "main.tex"
OUT = ROOT / "final_report_preview.pdf"

PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,margin=1.9cm,columnsep=0.7cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,microtype}
\usepackage[T1]{fontenc}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage{titlesec}
\graphicspath{{figures/}{./}}
\titleformat{\section}{\normalfont\large\bfseries}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection.}{0.5em}{}
\setlength{\parskip}{0pt}
\renewcommand{\arraystretch}{1.05}

\title{\vspace{-1.2cm}\bfseries %(title)s}
\author{%(author)s\\[2pt]
\normalsize %(affil)s\\
\normalsize \texttt{%(email)s}}
\date{\normalsize %(date)s}

\begin{document}
\twocolumn[
  \begin{@twocolumnfalse}
  \maketitle
  \begin{abstract}
  \noindent %(abstract)s
  \end{abstract}
  \vspace{0.6em}
  \end{@twocolumnfalse}
]
"""


def extract(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.S)
    return m.group(1).strip() if m else default


def main() -> None:
    src = SRC.read_text()

    fields = {
        "title": extract(r"\\title\{(.+?)\}\s*\n\s*\n", src) or extract(r"\\title\{(.+?)\}", src),
        "author": extract(r"\\author\[[^\]]*\]\{(.+?)\}", src, "Po-Hsun Lai").replace(",", ""),
        "affil": extract(r"\\affil\[1\]\{(.+?)\}", src, ""),
        "email": extract(r"\\affil\[\*\]\{(.+?)\}", src, ""),
        "abstract": extract(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", src),
        "date": "September 2026",
    }
    body = src.split(r"\begin{document}", 1)[1]
    body = body.replace(r"\maketitle", "", 1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "preview.tex").write_text(PREAMBLE % fields + body)
        shutil.copytree(ROOT / "overleaf" / "figures", tmp / "figures")

        for i in range(3):
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
                cwd=tmp, capture_output=True, text=True)
            if proc.returncode != 0 and i == 0:
                errs = [ln for ln in proc.stdout.splitlines() if ln.startswith("!")]
                raise SystemExit("pdflatex failed:\n" + "\n".join(errs[:15]))

        log = (tmp / "preview.log").read_text(errors="ignore")
        bad = [ln for ln in log.splitlines() if "undefined" in ln.lower()]
        shutil.copy(tmp / "preview.pdf", OUT)

    pages = subprocess.run(
        ["python3", "-c",
         f"import pymupdf;print(pymupdf.open('{OUT}').page_count)"],
        capture_output=True, text=True).stdout.strip()
    print(f"wrote {OUT}  ({pages} pages)")
    if bad:
        print("unresolved references:")
        for ln in bad[:10]:
            print("  " + ln)
    else:
        print("all references and citations resolved")


if __name__ == "__main__":
    main()
