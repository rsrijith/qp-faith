#!/usr/bin/env python3
"""Minimal Markdown -> LaTeX -> PDF converter for editor letters / responses.
Handles: #/##/### headings, > blockquotes (with nested - bullets and blank-line
paragraph breaks), --- rules, - and 1. lists, **bold**, *italic*, `code`, and
LaTeX/unicode escaping. Not a general Markdown engine; tuned for these letters.

Usage: python3 scripts/md_to_pdf_letter.py <path/to/file.md>
Writes <file>.tex next to it and compiles <file>.pdf with pdflatex.
"""
import sys, os, re, subprocess

def esc(s):
    # unicode -> LaTeX first
    u = {"–": "--", "—": "---", "×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$",
         "≈": r"$\approx$", "…": r"\ldots{}", "“": "``", "”": "''", "‘": "`",
         "’": "'", " ": "~", "→": r"$\rightarrow$", "•": r"\textbullet{}"}
    for a, b in u.items():
        s = s.replace(a, b)
    # escape LaTeX specials
    out = []
    for ch in s:
        if ch in "&%#_{}$":
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == "\\":
            out.append("\\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)

def inline(s):
    # tokenize bold/italic/code before escaping so markers survive
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: "\x01" + m.group(1) + "\x02", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", lambda m: "\x03" + m.group(1) + "\x04", s)
    s = re.sub(r"`(.+?)`", lambda m: "\x05" + m.group(1) + "\x06", s)
    s = esc(s)
    s = (s.replace("\x01", r"\textbf{").replace("\x02", "}")
           .replace("\x03", r"\textit{").replace("\x04", "}")
           .replace("\x05", r"\texttt{").replace("\x06", "}"))
    return s

def main():
    src = sys.argv[1]
    lines = open(src, encoding="utf-8").read().split("\n")
    out = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=1in]{geometry}",
        r"\usepackage{parskip}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{enumitem}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\hypersetup{hidelinks}",
        r"\setlength{\emergencystretch}{3em}",
        r"\begin{document}",
    ]
    i, n = 0, len(lines)
    def flush_para(buf):
        if buf:
            out.append(inline(" ".join(buf)))
            out.append("")
            buf.clear()
    para = []
    while i < n:
        ln = lines[i]
        if ln.startswith("# "):
            flush_para(para)
            out.append(r"\begin{center}\LARGE\bfseries " + inline(ln[2:]) + r"\end{center}")
            out.append("")
        elif ln.startswith("## "):
            flush_para(para)
            out.append(r"\vspace{0.6em}\noindent{\large\bfseries " + inline(ln[3:]) + r"}\par\vspace{0.2em}")
        elif ln.startswith("### "):
            flush_para(para)
            out.append(r"\noindent{\bfseries " + inline(ln[4:]) + r"}\par")
        elif ln.strip() == "---":
            flush_para(para)
            out.append(r"\vspace{0.4em}\hrule\vspace{0.4em}")
        elif ln.startswith(">"):
            flush_para(para)
            block = []
            while i < n and lines[i].startswith(">"):
                block.append(lines[i][1:].lstrip())
                i += 1
            out.append(r"\begin{quote}")
            sub, in_list = [], False
            def flush_sub():
                if sub:
                    out.append(inline(" ".join(sub)))
                    sub.clear()
            for b in block:
                if b.startswith("- "):
                    flush_sub()
                    if not in_list:
                        out.append(r"\begin{itemize}[leftmargin=1.2em,itemsep=1pt]")
                        in_list = True
                    out.append(r"\item " + inline(b[2:]))
                elif b == "":
                    flush_sub()
                    if in_list:
                        out.append(r"\end{itemize}"); in_list = False
                    out.append("")
                else:
                    if in_list:
                        out.append(r"\end{itemize}"); in_list = False
                    sub.append(b)
            flush_sub()
            if in_list:
                out.append(r"\end{itemize}")
            out.append(r"\end{quote}")
            continue
        elif re.match(r"^\d+\.\s", ln):
            flush_para(para)
            out.append(r"\begin{enumerate}[leftmargin=1.5em,itemsep=1pt]")
            while i < n and re.match(r"^\d+\.\s", lines[i]):
                out.append(r"\item " + inline(re.sub(r"^\d+\.\s", "", lines[i])))
                i += 1
            out.append(r"\end{enumerate}")
            continue
        elif ln.startswith("- "):
            flush_para(para)
            out.append(r"\begin{itemize}[leftmargin=1.5em,itemsep=1pt]")
            while i < n and lines[i].startswith("- "):
                out.append(r"\item " + inline(lines[i][2:]))
                i += 1
            out.append(r"\end{itemize}")
            continue
        elif ln.strip() == "":
            flush_para(para)
        else:
            para.append(ln)
        i += 1
    flush_para(para)
    out.append(r"\end{document}")
    tex = os.path.splitext(src)[0] + ".tex"
    open(tex, "w", encoding="utf-8").write("\n".join(out))
    d = os.path.dirname(os.path.abspath(tex))
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", os.path.basename(tex)],
                       cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", os.path.splitext(src)[0] + ".pdf")

if __name__ == "__main__":
    main()
