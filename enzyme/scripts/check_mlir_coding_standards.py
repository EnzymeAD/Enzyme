#!/usr/bin/env python3
"""Check MLIR C++ and lit test files for non-AST MLIR coding standards.

C++ AST-level MLIR checks are enforced via `clang-tidy`
(`enzyme/Enzyme/MLIR/.clang-tidy`, `enzyme/Enzyme/MLIR/Passes/.clang-tidy`, and
`enzyme/scripts/run_clang_tidy.py`). This linter enforces the 5 remaining
comment-syntax and `.mlir` lit-test rules that `clang-query` AST matchers cannot
inspect:

- Whole-tree checks (`whole_tree=True`):
  1. `mlir-lit-stderr-devnull` (`.mlir` files)
  2. `mlir-lit-negative-func-name` (`.mlir` files)
- Diff-scoped checks (`whole_tree=False`):
  3. `mlir-lit-pinned-ssa` (`.mlir` files)
  4. `mlir-lit-closing-brace` (`.mlir` files)
  5. `mlir-doxygen-comment` (`.cpp`/`.cc`/`.h`/`.hpp` `//` vs `///` doc comments)

Suppress an intentional exception with `// NOLINT(<check>)` on the flagged line
or `// NOLINTNEXTLINE(<check>)` on the preceding line.
"""

import argparse
from dataclasses import dataclass
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set

CPP_SUFFIXES = (".cpp", ".cc", ".h", ".hpp")
MLIR_TEST_SUFFIXES = (".mlir",)
ALL_SUFFIXES = CPP_SUFFIXES + MLIR_TEST_SUFFIXES
SKIP_DIRS = {".git", "build", "third_party", "external"}


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
            # C++ raw string literal: R"delim(...)delim"
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
# Regex definitions for non-AST MLIR doc-comment and `.mlir` lit checks
# ---------------------------------------------------------------------------

# Non-Doxygen `//` doc comments immediately preceding top-level/member
# function or class/struct declarations in MLIR headers/sources.
DOC_COMMENT_TARGET_RE = re.compile(
    r"^\s*(?:template\s*<[^>]+>\s*)?"
    r"(?:(?:static|virtual|explicit|inline|constexpr)\s+)*"
    r"(?:class|struct|(?:[\w:]+(?:<[^;{}]+>)?\s+[*&]?\s*[A-Za-z_]\w*\s*\())"
)

# MLIR `.mlir` lit test checks.
MLIR_CHECK_LINE_RE = re.compile(
    r"^\s*//\s*CHECK(?:-[A-Z]+)?(?:\{[A-Z]+\})?\s*:(.*)$"
)
MLIR_PINNED_SSA_RE = re.compile(r"(?<!\[)%(\d+|arg\d+)\b")
MLIR_CLOSING_BRACE_CHECK_RE = re.compile(r"^\s*//\s*CHECK-NEXT\s*:\s*\}\s*$")
MLIR_RUN_LINE_RE = re.compile(r"^\s*//\s*RUN\s*:(.*)$")
STDERR_DEVNULL_RE = re.compile(r"2>\s*/dev/null")
NEGATIVE_FUNC_SUFFIX_RE = re.compile(
    r"\bfunc\.func\b[^@\n]*@(\w+_(?:fail|fails|error|invalid))\b"
)


CHECK_SPECS: List[CheckSpec] = [
    CheckSpec(
        "mlir-lit-stderr-devnull",
        True,
        "Do not redirect stderr to `/dev/null` (`2>/dev/null`) on lit `RUN:` lines.",
    ),
    CheckSpec(
        "mlir-lit-negative-func-name",
        True,
        "Prefix negative diagnostic test functions with `@negative_...` (or `@invalid_...`) instead of a `@..._fail` suffix.",
    ),
    CheckSpec(
        "mlir-lit-pinned-ssa",
        False,
        "Do not hardcode numbered SSA names (`%0`, `%arg0`) in `CHECK` lines; capture with `%[[NAME:.*]]`.",
    ),
    CheckSpec(
        "mlir-lit-closing-brace",
        False,
        "Avoid `CHECK-NEXT: }` lines that only assert `FuncOp`/module printer boilerplate.",
    ),
    CheckSpec(
        "mlir-doxygen-comment",
        False,
        "Use `///` Doxygen comments (not plain `//`) for doc comments on functions, classes, and structs in MLIR.",
    ),
]

SPEC_BY_NAME: Dict[str, CheckSpec] = {spec.name: spec for spec in CHECK_SPECS}


def check_cpp_source(path: str, raw: str) -> List[Finding]:
    findings: List[Finding] = []
    raw_lines = raw.splitlines()
    clean_lines = strip_noise(raw).splitlines()

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

    # Track brace depth per line so doc-comment check only inspects top-level or
    # class scope, never inline comments inside function bodies.
    line_depths: List[int] = []
    in_class_scope: List[bool] = []
    depth = 0
    class_depths: List[int] = []
    class_kw_re = re.compile(r"\b(?:class|struct)\s+[A-Za-z_]\w*[^;]*\{")
    for line in clean_lines:
        line_depths.append(depth)
        in_class_scope.append(bool(class_depths))
        opens_class = bool(class_kw_re.search(line))
        for ch in line:
            if ch == "{":
                depth += 1
                if opens_class:
                    class_depths.append(depth)
                    opens_class = False
            elif ch == "}":
                if class_depths and class_depths[-1] == depth:
                    class_depths.pop()
                depth = max(0, depth - 1)

    for idx, line in enumerate(clean_lines):
        if idx > 0 and (line_depths[idx] == 0 or in_class_scope[idx]):
            if DOC_COMMENT_TARGET_RE.match(line) and not line.lstrip().startswith("return "):
                prev_raw = raw_lines[idx - 1].lstrip()
                if (
                    prev_raw.startswith("//")
                    and not prev_raw.startswith("///")
                    and not prev_raw.startswith("//!")
                    and not any(
                        tag in prev_raw
                        for tag in (
                            "NOLINT",
                            "clang-format",
                            "===--",
                            "---==",
                            "TODO",
                            "FIXME",
                            "NOTE",
                        )
                    )
                ):
                    add(
                        idx - 1,
                        "mlir-doxygen-comment",
                        SPEC_BY_NAME["mlir-doxygen-comment"].description,
                    )

    return findings


def check_mlir_lit_test(path: str, raw: str) -> List[Finding]:
    findings: List[Finding] = []
    raw_lines = raw.splitlines()
    has_verify_diagnostics = "-verify-diagnostics" in raw or "--verify-diagnostics" in raw

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
        m_run = MLIR_RUN_LINE_RE.match(line)
        if m_run and STDERR_DEVNULL_RE.search(m_run.group(1)):
            add(
                idx,
                "mlir-lit-stderr-devnull",
                SPEC_BY_NAME["mlir-lit-stderr-devnull"].description,
            )

        if MLIR_CLOSING_BRACE_CHECK_RE.match(line):
            add(
                idx,
                "mlir-lit-closing-brace",
                SPEC_BY_NAME["mlir-lit-closing-brace"].description,
            )

        m_check = MLIR_CHECK_LINE_RE.match(line)
        if m_check:
            body = m_check.group(1)
            # Strip FileCheck `[[...]]` variable blocks and `{{...}}` regexes before scanning.
            scrubbed = re.sub(r"\[\[.*?\]\]|\{\{.*?\}\}", "", body)
            m_ssa = MLIR_PINNED_SSA_RE.search(scrubbed)
            if m_ssa:
                add(
                    idx,
                    "mlir-lit-pinned-ssa",
                    f"Hardcoded SSA value `%{m_ssa.group(1)}` in CHECK line; capture with `%[[NAME:.*]]`.",
                )

        if has_verify_diagnostics:
            m_neg = NEGATIVE_FUNC_SUFFIX_RE.search(line)
            if m_neg and not m_neg.group(1).startswith(("negative_", "invalid_", "no_")):
                add(
                    idx,
                    "mlir-lit-negative-func-name",
                    f"Negative test function `@{m_neg.group(1)}` should use a `@negative_...` prefix instead of a `..._fail` suffix.",
                )

    return findings


def check_text(path: str, raw: str) -> List[Finding]:
    if path.endswith(MLIR_TEST_SUFFIXES):
        return check_mlir_lit_test(path, raw)
    if path.endswith(CPP_SUFFIXES):
        return check_cpp_source(path, raw)
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
                f"title=MLIR coding standard ({f.check})::{f.message}"
            )

    if all_findings:
        print(
            f"\n{len(all_findings)} MLIR coding standard violation(s) found. "
            f"Fix the violation or add `// NOLINT(<check>)` for an intentional exception.",
            file=sys.stderr,
        )
        return 1

    print(f"checked {len(files)} MLIR files: no coding standard violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
