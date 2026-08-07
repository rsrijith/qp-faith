#!/usr/bin/env python3
"""Produce the 'changes marked' manuscript for the IP&M revision.
latexdiff(submitted baseline, revised main) with BLUE additions only and
deletions suppressed (no red strike-throughs, per author preference).
Pure deletions are invisible in this mode by design; the Response to the
Editor documents any material deletion in words.

Run from the repo root:  python3 scripts/make_marked_diff.py
Then in latex_ipm/:  pdflatex main_diff && bibtex main_diff && pdflatex main_diff x2
"""
import subprocess, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
LATEX = os.path.join(HERE, "..", "latex_ipm")
OLD = os.path.join(LATEX, "main_OLD.tex")      # submitted baseline (no line numbers)
NEW = os.path.join(LATEX, "main.tex")          # revised manuscript
DIFF = os.path.join(LATEX, "main_diff.tex")

subprocess.run(["latexdiff", "--encoding=utf8", OLD, NEW],
               stdout=open(DIFF, "w"), check=True)

t = open(DIFF, encoding="utf-8").read()

# Blue additions only; suppress every deletion marker (text + float variants).
override = (
    "\\renewcommand{\\DIFadd}[1]{{\\protect\\color{blue}#1}}%\n"
    "\\renewcommand{\\DIFdel}[1]{}%\n"
    "\\providecommand{\\DIFaddFL}[1]{}\\renewcommand{\\DIFaddFL}[1]{{\\protect\\color{blue}#1}}%\n"
    "\\providecommand{\\DIFdelFL}[1]{}\\renewcommand{\\DIFdelFL}[1]{}%\n"
    "\\begin{document}"
)
assert t.count("\\begin{document}") == 1, "expected exactly one \\begin{document}"
t = t.replace("\\begin{document}", override, 1)
open(DIFF, "w", encoding="utf-8").write(t)
print("wrote", DIFF, "(blue additions only, deletions suppressed)")
