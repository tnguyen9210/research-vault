#!/usr/bin/env python3
"""Check that wiki math survives GitHub's markdown renderer.

Usage:
    check_math.py [FILE ...]        (default: every wiki page)
    check_math.py --summary [FILE ...]

GitHub runs markdown over `$...$` and `$$...$$` math before
MathJax sees it, so backslash escapes (`\\{ \\, \\; \\!`), emphasis
pairs (`*`, some `_`), links (`[..](..)`) and block syntax (a
line holding only `=` or `-`) silently change or break the
math. This script renders each page through GitHub's own
markdown API and compares every math span in the source with
what GitHub hands to MathJax. A span is FAITHFUL when its
content arrives unchanged (whitespace aside).

Fenced ```math blocks and $`...`$ inline spans are understood
too; GitHub passes both through untouched.

Requires an authenticated `gh` CLI (the markdown API allows
60 unauthenticated calls an hour, fewer than one full run).
Exit status is 1 when any span is wrong, 2 on an API failure,
which is never reported as a pass.
"""

import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys
import time

REPO = "tnguyen9210/research-vault"
WIKI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "research_vault", "wiki")
SKIP = {"log.md"}
RETRIES = 5
LOOKAHEAD = 40
MATH_EL = re.compile(
    r'<math-renderer class="js-(inline|display)-math"[^>]*>'
    r'(.*?)</math-renderer>', re.S)


def render(text):
    """GitHub's HTML for `text`; raises rather than return ''."""
    payload = json.dumps({"text": text, "mode": "gfm",
                          "context": REPO})
    err = ""
    for attempt in range(RETRIES):
        out = subprocess.run(
            ["gh", "api", "-X", "POST", "/markdown", "--input", "-"],
            input=payload, capture_output=True, text=True)
        if out.returncode == 0 and out.stdout:
            return out.stdout
        err = out.stderr.strip()
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"markdown API failed: {err[:200]}")


def served(content):
    """What MathJax receives, minus one known harmless rewrite.

    In a ```math fence GitHub turns a row break at the end of a
    line into a triple backslash. MathJax reads the extra one as
    a no-break space opening the next row, which shifts every
    row of an `aligned` block equally (checked 2026-09-22 against
    MathJax 3.2.1's MathML), so it is not counted as a change.
    """
    c = html.unescape(html.unescape(content))
    return re.sub(r"(?<!\\)\\\\\\\n", lambda m: "\\\\\n", c)


def squash(s):
    return re.sub(r"\s+", "", s)


def source_spans(src):
    """[(kind, content, line)] for every math span, in order."""
    spans = []

    def line_of(pos):
        return src.count("\n", 0, pos) + 1

    # Blank out everything that is not math, keeping offsets.
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))

    work = re.sub(r"<!--.*?-->", blank, src, flags=re.S)
    for m in re.finditer(r"^```math[ \t]*\n(.*?)^```", work,
                         re.S | re.M):
        spans.append((m.start(), "display", m.group(1)))
    work = re.sub(r"^```.*?^```", blank, work, flags=re.S | re.M)
    for m in re.finditer(r"\$`([^`\n]+)`\$", work):
        spans.append((m.start(), "inline", m.group(1)))
    work = re.sub(r"\$`[^`\n]+`\$", blank, work)
    work = re.sub(r"`[^`\n]*`", blank, work)
    for m in re.finditer(r"\$\$(.+?)\$\$", work, re.S):
        spans.append((m.start(), "display", m.group(1)))
    work = re.sub(r"\$\$(.+?)\$\$", blank, work, flags=re.S)
    for m in re.finditer(r"(?<!\\)\$([^$\n]+?)\$", work):
        spans.append((m.start(), "inline", m.group(1)))
    return [(kind, content, line_of(pos))
            for pos, kind, content in sorted(spans)]


def likely_cause(kind, content, src, line):
    """Best guess at why a span is wrong, for the report only."""
    lines = src.split("\n")
    if kind == "display" and "$$" in lines[line - 1]:
        before = lines[line - 2] if line >= 2 else ""
        end = line - 1 + content.count("\n")
        after = lines[end + 1] if end + 1 < len(lines) else ""
        if before.strip() or after.strip():
            return "delimiter not on its own paragraph"
    body = content.replace("\\\\", "")
    here = lines[line - 1]
    if kind == "inline" and re.search(
            "[A-Za-z0-9‐-—-]\\$" + re.escape(content), here):
        return "opening `$` glued to a letter, digit or dash"
    if kind == "inline" and content.endswith(")") \
            and content + "$)" in here:
        return "span ending in `)` followed by `)`"
    if re.search(re.escape(content) + r"`?\$[*_](?![*_])", here):
        return "closing `$` touches a closing `*`/`_`"
    if re.search(r"(?m)^\s*([=+\-*])\s*$|^\s*([-+*>#]|\d+\.)\s",
                 content) and kind == "display":
        return "line read as markdown block syntax"
    if re.search(r"\\[!-/:-@\[-`{-~]", body):
        return "backslash escape stripped"
    if "*" in content:
        return "`*` read as emphasis"
    if "](" in content:
        return "`[..](..)` read as a link"
    return "collateral: another span in its paragraph broke"


def check(path):
    src = open(path).read()
    spans = source_spans(src)
    if not spans:
        return 0, []
    got = [squash(served(c)).strip("$`")
           for _, c in MATH_EL.findall(render(src))]
    # Match in document order, so a failure is blamed on the span
    # that actually failed rather than on a later lookalike.
    wrong, j = [], 0
    for kind, content, line in spans:
        key = squash(content)
        hit = next((i for i in range(j, min(j + LOOKAHEAD, len(got)))
                    if got[i] == key), None)
        if hit is not None:
            j = hit + 1
        else:
            wrong.append((line, kind,
                          likely_cause(kind, content, src, line),
                          " ".join(content.split())[:70]))
    return len(spans), wrong


def main():
    ap = argparse.ArgumentParser(
        description="Check wiki math against GitHub's renderer.")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--summary", action="store_true",
                    help="one line per failing file, no spans")
    args = ap.parse_args()
    files = args.files or sorted(
        f for f in glob.glob(os.path.join(WIKI, "**", "*.md"),
                             recursive=True)
        if os.path.basename(f) not in SKIP)

    n_spans = n_wrong = n_files = 0
    for f in files:
        try:
            total, wrong = check(f)
        except RuntimeError as e:
            print(f"ERROR    {f}: {e}")
            sys.exit(2)
        n_spans += total
        n_wrong += len(wrong)
        if wrong:
            n_files += 1
            rel = os.path.relpath(f, WIKI)
            name = f if rel.startswith("..") else rel
            print(f"{len(wrong):4d}/{total:<4d} wrong  {name}")
            if not args.summary:
                for line, kind, cause, text in wrong:
                    print(f"      L{line:<5d}{kind:8s}{cause}")
                    print(f"             {text}")
        time.sleep(0.3)
    print(f"done: {n_wrong}/{n_spans} spans wrong "
          f"in {n_files} of {len(files)} files")
    sys.exit(1 if n_wrong else 0)


if __name__ == "__main__":
    main()
