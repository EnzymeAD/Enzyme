#!/usr/bin/env python3
"""Check C++, TableGen, and LLVM lit test files for non-AST LLVM coding standards.

C++ AST-level LLVM checks are enforced via `clang-tidy` (`enzyme/.clang-tidy` and
`enzyme/scripts/run_clang_tidy.py`). This linter enforces the 6 remaining
text/preprocessor/lit-test checks that `clang-query` AST matchers cannot inspect
(or that compile away under `-DNDEBUG`):

- Whole-tree checks (`whole_tree=True`):
  1. `llvm-lit-stderr-devnull` (`.ll` files)
- Diff-scoped checks (`whole_tree=False`):
  2. `llvm-lit-pinned-ssa` (`.ll` files)
  3. `llvm-assert-false` (`.cpp`/`.cc`/`.h`/`.hpp` raw macro text)
  4. `llvm-assert-no-message` (`.cpp`/`.cc`/`.h`/`.hpp` raw macro text)
  5. `llvm-include-comment` (`.cpp`/`.cc`/`.h`/`.hpp` `#include` trailing comment)
  6. `llvm-non-ascii` (`.cpp`/`.cc`/`.h`/`.hpp`/`.td`/`.ll`)

Suppress an intentional exception with `// NOLINT(<check>)` (or
`; NOLINT(<check>)` in `.ll` files) on the flagged line, or
`NOLINTNEXTLINE(<check>)` on the preceding line.
"""

import argparse
from dataclasses import dataclass
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set

CPP_SUFFIXES = (".cpp", ".cc", ".h", ".hpp")
TD_SUFFIXES = (".td",)
LL_TEST_SUFFIXES = (".ll",)
ALL_SUFFIXES = CPP_SUFFIXES + TD_SUFFIXES + LL_TEST_SUFFIXES
SKIP_DIRS = {".git", "build", "third_party", "external", "SCEV"}


@dataclass(frozen=True)
class Finding:
    path: str
    lineno: int
    check: str
    message: str
    snippet: str


@dataclass(frozen=True)
class CheckSpec:
    name: str
    whole_tree: bool
    description: str


def strip_noise(src: str) -> str:
    """Blank out comments, strings, raw strings, and char literals, keeping offsets."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            for j in range(i, min(i + 2, n)):
                out[j] = " "
            i += 2
        elif (
            c == "R"
            and i + 1 < n
            and src[i + 1] == '"'
            and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_"))
        ):
            j = i + 2
            while j < n and src[j] not in '()\\\n\t "':
                j += 1
            if j < n and src[j] == "(":
                delim = src[i + 2 : j]
                end_seq = ")" + delim + '"'
                for k in range(i, j + 1):
                    out[k] = " "
                i = j + 1
                end_pos = src.find(end_seq, i)
                if end_pos == -1:
                    end_pos = n
                while i < end_pos:
                    if src[i] != "\n":
                        out[i] = " "
                    i += 1
                for k in range(end_pos, min(end_pos + len(end_seq), n)):
                    if src[k] != "\n":
                        out[k] = " "
                i = min(end_pos + len(end_seq), n)
            else:
                i += 1
        elif c in "\"'":
            quote = c
            out[i] = " "
            i += 1
            while i < n and src[i] != quote:
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n:
                    if src[i] != "\n":
                        out[i] = " "
                    i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def suppressed(raw_lines: List[str], idx: int, check: str) -> bool:
    """Return True if 0-based line `idx` carries a NOLINT marker for `check`."""
    same = re.compile(r"NOLINT\(\s*(?:\*|" + re.escape(check) + r")\s*\)")
    prev = re.compile(r"NOLINTNEXTLINE\(\s*(?:\*|" + re.escape(check) + r")\s*\)")
    if same.search(raw_lines[idx]):
        return True
    return idx > 0 and bool(prev.search(raw_lines[idx - 1]))


def iter_files(roots: Iterable[str]) -> Iterable[str]:
    for root in roots:
        if os.path.isfile(root):
            if root.endswith(ALL_SUFFIXES):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIRS and not d.startswith("bazel-")
            ]
            for name in sorted(filenames):
                if name.endswith(ALL_SUFFIXES):
                    yield os.path.join(dirpath, name)


# ---------------------------------------------------------------------------
# Regex definitions for non-AST LLVM C++, TableGen, and `.ll` checks
# ---------------------------------------------------------------------------

ASSERT_TOKEN_RE = re.compile(r"\bassert\s*\(")

# `assert(false ...)` / `assert(0 ...)` instead of `llvm_unreachable(...)`.
ASSERT_FALSE_RE = re.compile(r'\bassert\s*\(\s*(?:false|0|!"[^"]*")\b')

# Single-line `assert(...)` call on raw text so string message literals remain visible.
RAW_SINGLE_LINE_ASSERT_RE = re.compile(r"\bassert\s*\((.+)\)\s*;")

# `#include` lines with trailing explanatory comments (allow IWYU, clang-format, NOLINT).
INCLUDE_COMMENT_RE = re.compile(
    r'^\s*#\s*include\s+[<"][^>"]+[>"]\s*(//.*)$'
)
ALLOWED_INCLUDE_PRAGMA_RE = re.compile(
    r"^//\s*(?:IWYU\s+pragma:|clang-format\b|NOLINT\b)"
)

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# `.ll` lit test patterns.
LL_RUN_LINE_RE = re.compile(r"^\s*;\s*RUN\s*:(.*)$")
STDERR_DEVNULL_RE = re.compile(r"2>\s*/dev/null")
LL_CHECK_LINE_RE = re.compile(r"^\s*;\s*CHECK(?:-[A-Z]+)?\s*:(.*)$")
LL_PINNED_NUMERIC_SSA_RE = re.compile(r"(?<!\[)%(\d+)\b")
LL_ATTR_GROUP_RE = re.compile(r"(?<=\s)#\d+\b")


CHECK_SPECS: List[CheckSpec] = [
    CheckSpec(
        "llvm-lit-stderr-devnull",
        True,
        "Do not redirect stderr to `/dev/null` (`2>/dev/null`) on lit `RUN:` lines.",
    ),
    CheckSpec(
        "llvm-lit-pinned-ssa",
        False,
        "Avoid hardcoded numeric SSA values (`%0`, `%1`) or unstable attribute groups (`#0`) in LLVM IR `CHECK` lines.",
    ),
    CheckSpec(
        "llvm-assert-false",
        False,
        'Use `llvm_unreachable("reason")` instead of `assert(false ...)` / `assert(0 ...)`.',
    ),
    CheckSpec(
        "llvm-assert-no-message",
        False,
        'Include an explanatory message in assertions: `assert(Cond && "explanation")`.',
    ),
    CheckSpec(
        "llvm-include-comment",
        False,
        "Do not attach explanatory comments to `#include` lines (tool pragmas like `IWYU pragma` are allowed).",
    ),
    CheckSpec(
        "llvm-non-ascii",
        False,
        "Keep source, comments, and tests ASCII-only (use `--` and `->`, never Unicode dashes or arrows).",
    ),
]

SPEC_BY_NAME: Dict[str, CheckSpec] = {spec.name: spec for spec in CHECK_SPECS}


def check_cpp_or_td(path: str, raw: str) -> List[Finding]:
    findings: List[Finding] = []
    raw_lines = raw.splitlines()
    clean_lines = strip_noise(raw).splitlines()
    is_cpp = path.endswith(CPP_SUFFIXES)
    norm_path = path.replace("\\", "/")
    is_lit_test_cpp = "/test/" in norm_path or norm_path.startswith("test/")

    def add(idx: int, check: str, msg: str) -> None:
        if suppressed(raw_lines, idx, check):
            return
        findings.append(
            Finding(
                path=path,
                lineno=idx + 1,
                check=check,
                message=msg,
                snippet=raw_lines[idx].strip(),
            )
        )

    # Check non-ASCII across C++ / TableGen files.
    for idx, rline in enumerate(raw_lines):
        if NON_ASCII_RE.search(rline):
            add(idx, "llvm-non-ascii", SPEC_BY_NAME["llvm-non-ascii"].description)

    if not is_cpp:
        return findings

    for idx, line in enumerate(clean_lines):
        rline = raw_lines[idx]

        if ASSERT_TOKEN_RE.search(line):
            if ASSERT_FALSE_RE.search(rline):
                add(
                    idx,
                    "llvm-assert-false",
                    SPEC_BY_NAME["llvm-assert-false"].description,
                )
            elif not is_lit_test_cpp:
                m_assert = RAW_SINGLE_LINE_ASSERT_RE.search(rline)
                if m_assert:
                    cond_text = m_assert.group(1)
                    if '"' not in cond_text:
                        add(
                            idx,
                            "llvm-assert-no-message",
                            SPEC_BY_NAME["llvm-assert-no-message"].description,
                        )

        if line.lstrip().startswith("#"):
            m_inc = INCLUDE_COMMENT_RE.match(rline)
            if m_inc and not ALLOWED_INCLUDE_PRAGMA_RE.match(m_inc.group(1)):
                add(
                    idx,
                    "llvm-include-comment",
                    SPEC_BY_NAME["llvm-include-comment"].description,
                )

    return findings


def check_ll_lit_test(path: str, raw: str) -> List[Finding]:
    findings: List[Finding] = []
    raw_lines = raw.splitlines()

    def add(idx: int, check: str, msg: str) -> None:
        if suppressed(raw_lines, idx, check):
            return
        findings.append(
            Finding(
                path=path,
                lineno=idx + 1,
                check=check,
                message=msg,
                snippet=raw_lines[idx].strip(),
            )
        )

    for idx, line in enumerate(raw_lines):
        if NON_ASCII_RE.search(line):
            add(idx, "llvm-non-ascii", SPEC_BY_NAME["llvm-non-ascii"].description)

        m_run = LL_RUN_LINE_RE.match(line)
        if m_run and STDERR_DEVNULL_RE.search(m_run.group(1)):
            add(
                idx,
                "llvm-lit-stderr-devnull",
                SPEC_BY_NAME["llvm-lit-stderr-devnull"].description,
            )

        m_check = LL_CHECK_LINE_RE.match(line)
        if m_check:
            body = m_check.group(1)
            scrubbed = re.sub(r"\[\[.*?\]\]|\{\{.*?\}\}", "", body)
            m_ssa = LL_PINNED_NUMERIC_SSA_RE.search(scrubbed)
            if m_ssa:
                add(
                    idx,
                    "llvm-lit-pinned-ssa",
                    f"Hardcoded numeric SSA value `%{m_ssa.group(1)}` in CHECK line; use a FileCheck capture (`%[[VAR:.+]]`).",
                )
            elif LL_ATTR_GROUP_RE.search(scrubbed):
                add(
                    idx,
                    "llvm-lit-pinned-ssa",
                    "Hardcoded attribute group reference (e.g. `#0`) in CHECK line is unstable across LLVM versions; omit or wildcard it.",
                )

    return findings


def check_text(path: str, raw: str) -> List[Finding]:
    if path.endswith(LL_TEST_SUFFIXES):
        return check_ll_lit_test(path, raw)
    if path.endswith(CPP_SUFFIXES + TD_SUFFIXES):
        return check_cpp_or_td(path, raw)
    return []


def compute_changed_lines(diff_base: str, cwd: Optional[str] = None) -> Dict[str, Set[int]]:
    """Return mapping of repo-absolute file path -> set of 1-based added/modified line numbers."""
    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=cwd, text=True
    ).strip()
    diff_out = subprocess.check_output(
        [
            "git",
            "-c",
            "diff.noprefix=false",
            "diff",
            "--no-ext-diff",
            "--unified=0",
            "--no-color",
            diff_base,
        ],
        cwd=top,
        text=True,
    )
    changed: Dict[str, Set[int]] = {}
    current_file: Optional[str] = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff_out.splitlines():
        if line.startswith("+++ b/"):
            rel = line[6:]
            current_file = os.path.realpath(os.path.join(top, rel))
        elif line.startswith("+++ /dev/null"):
            current_file = None
        elif current_file and line.startswith("@@ "):
            m = hunk_re.match(line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                if count > 0:
                    changed.setdefault(current_file, set()).update(
                        range(start, start + count)
                    )
    return changed


def filter_findings(
    findings: Iterable[Finding],
    *,
    all_checks: bool = False,
    changed_lines: Optional[Dict[str, Set[int]]] = None,
    selected: Optional[Set[str]] = None,
    ignored: Optional[Set[str]] = None,
) -> List[Finding]:
    kept: List[Finding] = []
    for f in findings:
        if selected is not None and f.check not in selected:
            continue
        if ignored is not None and f.check in ignored:
            continue
        spec = SPEC_BY_NAME[f.check]
        if all_checks or spec.whole_tree:
            kept.append(f)
        elif changed_lines is not None:
            real = os.path.realpath(f.path)
            if f.lineno in changed_lines.get(real, set()):
                kept.append(f)
    return kept


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roots", nargs="+", help="directories or files to check")
    parser.add_argument(
        "--diff-base",
        help="git revision against which diff-scoped checks are enforced on added/modified lines",
    )
    parser.add_argument(
        "--all-checks",
        action="store_true",
        help="run both whole-tree and diff-scoped checks across every line of all scanned files",
    )
    parser.add_argument(
        "--select",
        help="comma-separated subset of check IDs to run",
    )
    parser.add_argument(
        "--ignore",
        help="comma-separated check IDs to skip",
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="also emit GitHub Actions error annotations",
    )
    args = parser.parse_args(argv)

    selected = set(args.select.split(",")) if args.select else None
    ignored = set(args.ignore.split(",")) if args.ignore else None

    changed_lines: Optional[Dict[str, Set[int]]] = None
    if args.diff_base and not args.all_checks:
        first_root = args.roots[0]
        git_cwd = (
            first_root
            if os.path.isdir(first_root)
            else os.path.dirname(os.path.abspath(first_root))
        )
        changed_lines = compute_changed_lines(args.diff_base, cwd=git_cwd)

    files = list(iter_files(args.roots))
    all_findings: List[Finding] = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        raw_findings = check_text(path, raw)
        all_findings.extend(
            filter_findings(
                raw_findings,
                all_checks=args.all_checks,
                changed_lines=changed_lines,
                selected=selected,
                ignored=ignored,
            )
        )

    for f in all_findings:
        print(f"{f.path}:{f.lineno}: error: [{f.check}] {f.message}\n    {f.snippet}")
        if args.github:
            print(
                f"::error file={f.path},line={f.lineno},"
                f"title=LLVM coding standard ({f.check})::{f.message}"
            )

    if all_findings:
        print(
            f"\n{len(all_findings)} LLVM coding standard violation(s) found. "
            f"Fix the violation or add `// NOLINT(<check>)` for an intentional exception.",
            file=sys.stderr,
        )
        return 1

    print(f"checked {len(files)} LLVM files: no coding standard violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
