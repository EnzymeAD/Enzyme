#!/usr/bin/env python3
"""Check C++, TableGen, and LLVM lit test files against LLVM coding standards.

This linter enforces statically checkable rules from:
- `.agents/skills/llvm-coding-standards/`
- `.agents/skills/llvm-comment-hygiene/`
- `.agents/skills/llvm-style-review/`

Checks are split into two tiers:
- Whole-tree checks (`whole_tree=True`): invariants that hold across the entire
  codebase and are always enforced on all scanned files.
- Diff-scoped checks (`whole_tree=False`): style, comment, and test-hygiene
  rules enforced on added/modified lines when `--diff-base <rev>` is supplied,
  or across all lines when `--all-checks` is passed.

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
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
# Regex definitions for LLVM C++, TableGen, and `.ll` checks
# ---------------------------------------------------------------------------

IOSTREAM_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*<\s*iostream\s*>")
IOSTREAM_SYMBOL_RE = re.compile(r"\bstd::(?:cout|cerr|clog|endl)\b")

RTTI_EXCEPTIONS_RE = re.compile(
    r"\bdynamic_cast\s*<|\bthrow\b|\btry\s*\{|\bcatch\s*\("
)

# Local variable, return type, or member of type Twine (never allowed; Twine is
# valid ONLY as `const Twine &` in a function parameter list).
TWINE_DECL_RE = re.compile(
    r"^\s*(?:const\s+)?(?:llvm::)?Twine\s+[A-Za-z_]\w*\s*(?:=|\(|;|\{)"
)

STD_SORT_RE = re.compile(r"\bstd::sort\s*\(")

VECTOR_BOOL_RE = re.compile(r"\bstd::vector\s*<\s*bool\s*(?:,|>)\s*")

# `SmallVector<T, N> &` in function parameters instead of `SmallVectorImpl<T> &`
# or `ArrayRef<T>`.
SMALLVECTOR_PARAM_RE = re.compile(
    r"(?:[(,]\s*)(?:const\s+)?(?:llvm::)?SmallVector\s*<[^<>]+,\s*\d+\s*>\s*&"
)

# `const std::string &` in function parameters instead of `StringRef`.
CONST_STD_STRING_PARAM_RE = re.compile(
    r"(?:[(,]\s*)const\s+std::string\s*&"
)

# `assert(false ...)` / `assert(0 ...)` instead of `llvm_unreachable(...)`.
ASSERT_FALSE_RE = re.compile(
    r"\bassert\s*\(\s*(?:false|0|!\"[^\"]*\")\b"
)

# Single-line `assert(...)` call on raw text so string message literals remain visible.
RAW_SINGLE_LINE_ASSERT_RE = re.compile(r"\bassert\s*\((.+)\)\s*;")

PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once\b")

# `#include` lines with trailing explanatory comments (allow IWYU, clang-format, NOLINT).
INCLUDE_COMMENT_RE = re.compile(
    r"^\s*#\s*include\s+[<\"][^>\"]+[>\"]\s*(//.*)$"
)
ALLOWED_INCLUDE_PRAGMA_RE = re.compile(
    r"^//\s*(?:IWYU\s+pragma:|clang-format\b|NOLINT\b)"
)

# Streaming `"\n"` to raw_ostream instead of `'\n'`.
RAW_OSTREAM_DOUBLE_QUOTE_NL_RE = re.compile(
    r"(?:\b(?:errs|outs|dbgs)\s*\(\s*\)|\bOS\b|\bos\b|\bss\b)[^\n;]*<<\s*\"\\n\""
)

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# Anonymous namespace free function detection helpers.
ANON_NS_OPEN_RE = re.compile(r"\bnamespace\s*\{")
TYPE_BODY_OPEN_RE = re.compile(r"\b(?:class|struct|union|enum)\b[^;{}]*\{")
FREE_FUNC_DEF_RE = re.compile(
    r"^\s*(?:template\s*<[^>]+>\s*)?"
    r"(?:(?:inline|constexpr)\s+)*"
    r"(?!static\b|class\b|struct\b|union\b|enum\b|namespace\b|using\b|typedef\b|if\b|for\b|while\b|switch\b)"
    r"[\w:]+(?:<[^;{}]+>)?[\s*&]+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$"
)

# `else` immediately after an unconditional `return` / `break` / `continue` /
# `llvm_unreachable` in the preceding `if` branch (excluding `if constexpr`).
UNCONDITIONAL_EXIT_RE = re.compile(
    r"^\s*(?:return\b|break\s*;|continue\s*;|llvm_unreachable\s*\()"
)
ELSE_AFTER_BRACE_RE = re.compile(r"^\s*\}\s*else\b")
STANDALONE_ELSE_RE = re.compile(r"^\s*else\b")
IF_HEADER_RE = re.compile(r"\bif\s*(constexpr\s*)?\(")

# `.ll` lit test patterns.
LL_RUN_LINE_RE = re.compile(r"^\s*;\s*RUN\s*:(.*)$")
STDERR_DEVNULL_RE = re.compile(r"2>\s*/dev/null")
LL_CHECK_LINE_RE = re.compile(r"^\s*;\s*CHECK(?:-[A-Z]+)?\s*:(.*)$")
LL_PINNED_NUMERIC_SSA_RE = re.compile(r"(?<!\[)%(\d+)\b")
LL_ATTR_GROUP_RE = re.compile(r"(?<=\s)#\d+\b")


CHECK_SPECS: List[CheckSpec] = [
    CheckSpec(
        "llvm-no-rtti-exceptions",
        True,
        "Do not use C++ RTTI (`dynamic_cast`) or exceptions (`try`/`catch`/`throw`); use `isa`/`cast`/`dyn_cast` and `llvm::Error`/`LogicalResult`.",
    ),
    CheckSpec(
        "llvm-twine-local",
        True,
        "Never store `Twine` in a local variable or field (dangling temporary references); use `Twine` only as a `const Twine &` parameter.",
    ),
    CheckSpec(
        "llvm-std-sort",
        True,
        "Use `llvm::sort(...)` instead of `std::sort(...)`.",
    ),
    CheckSpec(
        "llvm-lit-stderr-devnull",
        True,
        "Do not redirect stderr to `/dev/null` (`2>/dev/null`) on lit `RUN:` lines.",
    ),
    CheckSpec(
        "llvm-no-iostream",
        True,
        "Do not use `<iostream>` (`std::cout`, `std::cerr`, `std::endl`); use `llvm/Support/raw_ostream.h` (`llvm::outs()`, `llvm::errs()`, `'\\n'`).",
    ),
    CheckSpec(
        "llvm-vector-bool",
        False,
        "Do not use `std::vector<bool>`; use `llvm::BitVector` or `llvm::SmallBitVector`.",
    ),
    CheckSpec(
        "llvm-smallvector-param",
        False,
        "Do not pin inline size `SmallVector<T, N> &` in function parameters; use `SmallVectorImpl<T> &` (mutating) or `ArrayRef<T>` (read-only).",
    ),
    CheckSpec(
        "llvm-const-std-string-param",
        False,
        "Prefer `llvm::StringRef` (by value) over `const std::string &` in function parameters.",
    ),
    CheckSpec(
        "llvm-assert-false",
        False,
        "Use `llvm_unreachable(\"reason\")` instead of `assert(false ...)` / `assert(0 ...)`.",
    ),
    CheckSpec(
        "llvm-assert-no-message",
        False,
        "Include an explanatory message in assertions: `assert(Cond && \"explanation\")`.",
    ),
    CheckSpec(
        "llvm-pragma-once",
        False,
        "Use standard `#ifndef ... #define ... #endif` header guards instead of `#pragma once`.",
    ),
    CheckSpec(
        "llvm-else-after-return",
        False,
        "Do not use `else` or `else if` after an unconditional `return`, `break`, `continue`, or `llvm_unreachable`.",
    ),
    CheckSpec(
        "llvm-anon-namespace-func",
        False,
        "Declare file-local free functions `static` outside anonymous namespaces; reserve `namespace { ... }` for `class`/`struct` types.",
    ),
    CheckSpec(
        "llvm-include-comment",
        False,
        "Do not attach explanatory comments to `#include` lines (tool pragmas like `IWYU pragma` are allowed).",
    ),
    CheckSpec(
        "llvm-ostream-string-newline",
        False,
        "Prefer `'\\n'` over `\"\\n\"` when streaming a single newline to `raw_ostream`.",
    ),
    CheckSpec(
        "llvm-non-ascii",
        False,
        "Keep source, comments, and tests ASCII-only (use `--` and `->`, never Unicode dashes or arrows).",
    ),
    CheckSpec(
        "llvm-lit-pinned-ssa",
        False,
        "Avoid hardcoded numeric SSA values (`%0`, `%1`) or unstable attribute groups (`#0`) in LLVM IR `CHECK` lines.",
    ),
]

SPEC_BY_NAME: Dict[str, CheckSpec] = {spec.name: spec for spec in CHECK_SPECS}


def _find_matching_brace(clean: str, open_pos: int) -> int:
    depth = 0
    for pos in range(open_pos, len(clean)):
        if clean[pos] == "{":
            depth += 1
        elif clean[pos] == "}":
            depth -= 1
            if depth == 0:
                return pos
    return len(clean) - 1


def check_cpp_or_td(path: str, raw: str) -> List[Finding]:
    findings: List[Finding] = []
    raw_lines = raw.splitlines()
    clean = strip_noise(raw)
    clean_lines = clean.splitlines()
    is_cpp = path.endswith(CPP_SUFFIXES)
    is_header = path.endswith((".h", ".hpp"))
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

    # Compute anonymous-namespace top-level line numbers (inside `namespace { ... }`,
    # at depth 1 of that namespace and NOT inside a nested `class`/`struct`).
    anon_top_lines: Set[int] = set()
    for m in ANON_NS_OPEN_RE.finditer(clean):
        open_pos = m.end() - 1
        close_pos = _find_matching_brace(clean, open_pos)
        start_line = clean.count("\n", 0, open_pos)  # 0-based
        end_line = clean.count("\n", 0, close_pos)
        rel_depth = 0
        for lno in range(start_line, end_line + 1):
            line_segment = clean_lines[lno]
            if lno == start_line:
                brace_idx = line_segment.find("{")
                if rel_depth == 1:
                    anon_top_lines.add(lno)
                for ch in line_segment[brace_idx:]:
                    if ch == "{":
                        rel_depth += 1
                    elif ch == "}":
                        rel_depth -= 1
                continue
            if rel_depth == 1:
                anon_top_lines.add(lno)
            for ch in line_segment:
                if ch == "{":
                    rel_depth += 1
                elif ch == "}":
                    rel_depth -= 1

    for idx, line in enumerate(clean_lines):
        rline = raw_lines[idx]

        if IOSTREAM_INCLUDE_RE.search(line) or IOSTREAM_SYMBOL_RE.search(line):
            if not is_lit_test_cpp:
                add(idx, "llvm-no-iostream", SPEC_BY_NAME["llvm-no-iostream"].description)

        if not is_lit_test_cpp and RTTI_EXCEPTIONS_RE.search(line):
            add(
                idx,
                "llvm-no-rtti-exceptions",
                SPEC_BY_NAME["llvm-no-rtti-exceptions"].description,
            )

        if TWINE_DECL_RE.search(line):
            add(idx, "llvm-twine-local", SPEC_BY_NAME["llvm-twine-local"].description)

        if not is_lit_test_cpp and STD_SORT_RE.search(line):
            add(idx, "llvm-std-sort", SPEC_BY_NAME["llvm-std-sort"].description)

        if VECTOR_BOOL_RE.search(line):
            add(idx, "llvm-vector-bool", SPEC_BY_NAME["llvm-vector-bool"].description)

        if SMALLVECTOR_PARAM_RE.search(line):
            add(
                idx,
                "llvm-smallvector-param",
                SPEC_BY_NAME["llvm-smallvector-param"].description,
            )

        if CONST_STD_STRING_PARAM_RE.search(line):
            add(
                idx,
                "llvm-const-std-string-param",
                SPEC_BY_NAME["llvm-const-std-string-param"].description,
            )

        if ASSERT_FALSE_RE.search(rline):
            add(idx, "llvm-assert-false", SPEC_BY_NAME["llvm-assert-false"].description)
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

        if is_header and PRAGMA_ONCE_RE.search(line):
            add(idx, "llvm-pragma-once", SPEC_BY_NAME["llvm-pragma-once"].description)

        m_inc = INCLUDE_COMMENT_RE.match(rline)
        if m_inc and not ALLOWED_INCLUDE_PRAGMA_RE.match(m_inc.group(1)):
            add(
                idx,
                "llvm-include-comment",
                SPEC_BY_NAME["llvm-include-comment"].description,
            )

        if RAW_OSTREAM_DOUBLE_QUOTE_NL_RE.search(rline):
            add(
                idx,
                "llvm-ostream-string-newline",
                SPEC_BY_NAME["llvm-ostream-string-newline"].description,
            )

        # `else` after unconditional `return`/`break`/`continue`/`llvm_unreachable`.
        if ELSE_AFTER_BRACE_RE.match(line) and idx >= 2:
            prev_stmt = clean_lines[idx - 1]
            # If the line before `return` is an unbraced `if (...)` / `for (...)` / `while (...)`,
            # then `return` is conditional, not unconditional for the enclosing `{ ... }` block!
            prev_ctrl_re = re.compile(r"\b(?:if|for|while)\s*\([^;{}]*\)\s*$")
            if UNCONDITIONAL_EXIT_RE.match(prev_stmt) and not prev_ctrl_re.search(
                clean_lines[idx - 2]
            ):
                # Walk backwards from the `}` on `line` to find its matching `{`.
                balance = 0
                open_line: Optional[int] = None
                close_col = line.find("}")
                for lno in range(idx, -1, -1):
                    seg = clean_lines[lno][: close_col + 1] if lno == idx else clean_lines[lno]
                    for ch in reversed(seg):
                        if ch == "}":
                            balance += 1
                        elif ch == "{":
                            balance -= 1
                            if balance == 0:
                                open_line = lno
                                break
                    if open_line is not None:
                        break
                if open_line is not None:
                    header_ctx = " ".join(
                        clean_lines[max(0, open_line - 2) : open_line + 1]
                    )
                    m_if = IF_HEADER_RE.search(header_ctx)
                    if m_if and not m_if.group(1):
                        add(
                            idx,
                            "llvm-else-after-return",
                            SPEC_BY_NAME["llvm-else-after-return"].description,
                        )
        elif STANDALONE_ELSE_RE.match(line) and idx >= 2:
            if UNCONDITIONAL_EXIT_RE.match(clean_lines[idx - 1]):
                m_if = IF_HEADER_RE.search(clean_lines[idx - 2])
                if m_if and not m_if.group(1):
                    add(
                        idx,
                        "llvm-else-after-return",
                        SPEC_BY_NAME["llvm-else-after-return"].description,
                    )

        # Anonymous-namespace free function definition.
        if idx in anon_top_lines and not TYPE_BODY_OPEN_RE.search(line):
            m_fn = FREE_FUNC_DEF_RE.match(line)
            if m_fn:
                has_body = "{" in line or (
                    idx + 1 < len(clean_lines)
                    and "{" in clean_lines[idx + 1]
                    and ";" not in clean_lines[idx + 1]
                )
                if has_body:
                    fn_name = m_fn.group(1)
                    add(
                        idx,
                        "llvm-anon-namespace-func",
                        f"Free function `{fn_name}` is inside `namespace {{ ... }}`; move it out and mark it `static`.",
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
        ["git", "diff", "--unified=0", "--no-color", diff_base],
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
