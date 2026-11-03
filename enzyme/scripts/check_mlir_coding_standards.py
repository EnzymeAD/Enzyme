#!/usr/bin/env python3
"""Check MLIR C++ and lit test files against MLIR coding standards.

This linter enforces statically checkable rules from
`.agents/skills/mlir-coding-standards/` across MLIR C++ sources (`.cpp`, `.cc`,
`.h`, `.hpp`, `.inc`) and MLIR lit tests (`.mlir`).

Checks are split into two tiers:
- Whole-tree checks (`whole_tree=True`): invariants that hold across the entire
  MLIR subtree and are always enforced on all scanned files.
- Diff-scoped checks (`whole_tree=False`): style and test-hygiene rules that
  are enforced on added/modified lines when `--diff-base <rev>` is supplied, or
  across all lines when `--all-checks` is passed.

Suppress an intentional exception with `// NOLINT(<check>)` on the flagged line
or `// NOLINTNEXTLINE(<check>)` on the preceding line.
"""

import argparse
from dataclasses import dataclass
import os
import re
import subprocess
import sys
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

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
                out[i] = " "
                for k in range(i + 1, j + 1):
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
    marker = r"NOLINT(?:NEXTLINE)?\(\s*(?:\*|" + re.escape(check) + r")\s*\)"
    same = re.compile(r"NOLINT\(\s*(?:\*|" + re.escape(check) + r")\s*\)")
    prev = re.compile(r"NOLINTNEXTLINE\(\s*(?:\*|" + re.escape(check) + r")\s*\)")
    if same.search(raw_lines[idx]):
        return True
    if idx > 0 and prev.search(raw_lines[idx - 1]):
        return True
    _ = marker
    return False


def find_matching_brace(clean: str, open_pos: int) -> int:
    """Given index of '{' in `clean`, return index of matching '}' or len(clean) - 1."""
    depth = 0
    for pos in range(open_pos, len(clean)):
        ch = clean[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return pos
    return len(clean) - 1


def offset_to_lineno(text: str, offset: int) -> int:
    """Convert 0-based character offset to 1-based line number."""
    return text.count("\n", 0, offset) + 1


def find_bodied_blocks(clean: str, header_re: re.Pattern) -> List[Tuple[int, int]]:
    """Return 1-based inclusive (start_line, end_line) ranges for matched `{ ... }` bodies."""
    ranges: List[Tuple[int, int]] = []
    for m in header_re.finditer(clean):
        open_pos = m.end() - 1
        if open_pos < 0 or clean[open_pos] != "{":
            continue
        close_pos = find_matching_brace(clean, open_pos)
        ranges.append(
            (offset_to_lineno(clean, open_pos), offset_to_lineno(clean, close_pos))
        )
    return ranges


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
# Regex definitions for MLIR C++ and lit checks
# ---------------------------------------------------------------------------

# 1. `const` on pointer IR units: Operation, Block, Region.
CONST_IR_PTR_RE = re.compile(
    r"\bconst\s+(?:mlir::)?(?:Operation|Block|Region)\s*\*"
    r"|\b(?:mlir::)?(?:Operation|Block|Region)\s+const\s*\*"
)

# 2. `const` on value-semantic IR handles (excluding pointer-to-const arrays
# such as `const mlir::Value *values` used by C APIs).
IR_HANDLE_TYPES = (
    r"(?:Value|Type|Attribute|Location|OpFoldResult|AffineMap|AffineExpr|"
    r"IntegerSet|ValueRange|TypeRange|[A-Z]\w*Op|[A-Z]\w*Attr|[A-Z]\w*Type)"
)
CONST_IR_VALUE_RE = re.compile(
    r"\bconst\s+(?:mlir::)?" + IR_HANDLE_TYPES + r"\b(?!\s*(?:\*|::))"
)

# 3. Passing small value-semantic MLIR handles by const-ref in parameters
# (excluding operator== / operator!= / operator< / DenseMapInfo).
PASS_IR_BY_CONST_REF_RE = re.compile(
    r"\bconst\s+(?:mlir::)?(?:Value|Type|Attribute|Location|OpFoldResult|"
    r"AffineMap|AffineExpr|IntegerSet|ValueRange|TypeRange|StringRef|"
    r"ArrayRef<[^>()]+>|[A-Z]\w*Op)\s*&"
)
COMPARISON_SIG_RE = re.compile(r"\b(?:operator\s*(?:==|!=|<|<=|>|>=)|DenseMapInfo)\b")

# 4. Builder / Rewriter passed by value in function signatures, or raw OpBuilder
# constructed inside `matchAndRewrite`.
BUILDER_TYPES = (
    r"(?:OpBuilder|RewriterBase|PatternRewriter|IRRewriter|"
    r"ConversionPatternRewriter|ImplicitLocOpBuilder)"
)
BUILDER_BY_VALUE_PARAM_RE = re.compile(
    r"(?:[(,]\s*)(?:const\s+)?(?:mlir::)?"
    + BUILDER_TYPES
    + r"\s+[A-Za-z_]\w*\s*(?:=[^{,)]+)?(?=[,)])"
)
RAW_OPBUILDER_IN_REWRITE_RE = re.compile(
    r"\b(?:mlir::)?(?:OpBuilder|IRRewriter)\s+[A-Za-z_]\w*\s*(?:\(|\{|=)"
)

# 5. Direct IR mutation inside `matchAndRewrite` bypassing `RewriterBase`.
MATCH_AND_REWRITE_HEADER_RE = re.compile(
    r"\bmatchAndRewrite\s*\([^;{}]*\)\s*(?:const\s*)?(?:override\s*)?(?:final\s*)?\{"
)
DIRECT_ERASE_OR_MOVE_RE = re.compile(
    r"(?:->|\.)\s*erase\s*\(\s*\)"
    r"|->\s*(?:moveBefore|moveAfter|remove)\s*\("
)
DIRECT_RAUW_RE = re.compile(r"(\w+)\s*(?:->|\.)\s*replaceAllUsesWith\s*\(")

# 6. Deprecated `builder.create<Op>(...)` instead of `Op::create(builder, ...)`.
BUILDER_CREATE_RE = re.compile(r"(?:->|\.)\s*create\s*<\s*[A-Za-z_][\w:]*\s*>\s*\(")

# 7. Non-local def-use chain inspection inside `Op::verify()` / `verifyRegions()`.
VERIFY_HEADER_RE = re.compile(
    r"::\s*verify(?:Regions)?\s*\(\s*\)\s*(?:const\s*)?\{"
)
NON_LOCAL_DEF_USE_RE = re.compile(
    r"(?:->|\.)\s*(?:getDefiningOp|getUsers|getUses|use_empty|hasOneUse)\s*(?:\(|<)"
)

# 8. Global `cl::opt` / `cl::list` inside MLIR pass implementations.
PASS_CL_OPT_RE = re.compile(r"\b(?:llvm::)?cl::(?:opt|list)\s*<")

# 9. Crashing/aborting inside `runOnOperation()` instead of `signalPassFailure()`.
RUN_ON_OP_HEADER_RE = re.compile(
    r"\brunOnOperation\s*\(\s*\)\s*(?:override\s*)?(?:final\s*)?\{"
)
PASS_ABORT_RE = re.compile(
    r"\b(?:llvm::)?report_fatal_error\s*\(|\b(?:std::)?(?:abort|exit|_Exit)\s*\("
)

# 10. Boolean error signaling (`FailureOr<bool>` or `bool matchAndRewrite/verify`).
BOOL_ERROR_RETURN_RE = re.compile(
    r"\b(?:mlir::)?FailureOr\s*<\s*bool\s*>"
    r"|\bbool\s+(?:\w+::)?(?:matchAndRewrite|verify|verifyRegions|verifySymbolUses)\s*\("
)

# 11. Explicit `static inline` in `.cpp` files.
STATIC_INLINE_CPP_RE = re.compile(r"\bstatic\s+inline\b|\binline\s+static\b")

# 12. `camelBack` variable naming in MLIR C++ files: flag obvious local variable
# declarations of common MLIR/LLVM types that use `UpperCamelCase` or `snake_case`.
CAMELBACK_LOCAL_DECL_RE = re.compile(
    r"^\s*(?:const\s+)?(?:mlir::|llvm::)?"
    r"(?:Value|Type|Attribute|Location|Operation\s*\*|Block\s*\*|Region\s*\*|"
    r"LogicalResult|SmallVector<[^;={}]+>|DenseMap<[^;={}]+>|ArrayRef<[^;={}]+>|"
    r"StringRef|OpFoldResult|auto\s*\*?|unsigned|int|size_t|int64_t|uint64_t|bool)"
    r"\s+([A-Za-z_]\w*)\s*(?:=|\{)"
)

# 13. Non-Doxygen `//` doc comments immediately preceding top-level/member
# function or class/struct declarations in MLIR headers/sources.
DOC_COMMENT_TARGET_RE = re.compile(
    r"^\s*(?:template\s*<[^>]+>\s*)?"
    r"(?:(?:static|virtual|explicit|inline|constexpr)\s+)*"
    r"(?:class|struct|(?:[\w:]+(?:<[^;{}]+>)?\s+[*&]?\s*[A-Za-z_]\w*\s*\())"
)

# 14. Overuse of `auto` when the type is not obvious from the RHS.
AUTO_NON_OBVIOUS_RE = re.compile(
    r"^\s*(?:const\s+)?auto\s*(?:&|\*)?\s+[A-Za-z_]\w*\s*=\s*"
    r"(?:[A-Za-z_]\w*(?:->|\.))*"
    r"(?:getOperand|getResult|getType|getLoc|getContext|getNumOperands|"
    r"getNumResults|getName|size|empty|front|back)\s*\("
)

# 15-18. MLIR `.mlir` lit test checks.
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
        "mlir-const-ir-pointer",
        True,
        "Do not qualify MLIR pointer IR units (`Operation *`, `Block *`, `Region *`) with `const`.",
    ),
    CheckSpec(
        "mlir-builder-by-value",
        True,
        "Pass `OpBuilder`/`RewriterBase` subclasses by reference (`&`), never by value.",
    ),
    CheckSpec(
        "mlir-verifier-non-local",
        True,
        "Op/region verifiers must check local properties only; do not walk unverified SSA def-use chains (`getDefiningOp`, `getUsers`, `getUses`).",
    ),
    CheckSpec(
        "mlir-bool-error-return",
        True,
        "Do not use `FailureOr<bool>` or return `bool` from `matchAndRewrite`/`verify*`; return `LogicalResult`.",
    ),
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
        "mlir-raw-builder-in-rewrite",
        False,
        "Do not construct a raw `OpBuilder`/`IRRewriter` inside `matchAndRewrite`; insert through the provided `PatternRewriter`.",
    ),
    CheckSpec(
        "mlir-direct-ir-mutation",
        False,
        "Do not mutate IR directly (`->erase()`, `replaceAllUsesWith`, `->moveBefore`) inside `matchAndRewrite`; go through `RewriterBase`.",
    ),
    CheckSpec(
        "mlir-pass-no-abort",
        False,
        "Passes must report errors via diagnostics + `signalPassFailure()`, never `report_fatal_error`/`abort`/`exit` inside `runOnOperation()`.",
    ),
    CheckSpec(
        "mlir-const-ir-value",
        False,
        "Do not qualify MLIR value-handle IR types (`Value`, `Type`, `Attribute`, `*Op`) with `const`.",
    ),
    CheckSpec(
        "mlir-pass-ir-by-value",
        False,
        "Pass small value-semantic MLIR handles (`Value`, `Type`, `Attribute`, `StringRef`, `ArrayRef`, `*Op`) by value, not `const &`.",
    ),
    CheckSpec(
        "mlir-op-create",
        False,
        "Prefer `Op::create(builder, loc, ...)` over deprecated `builder.create<Op>(loc, ...)`.",
    ),
    CheckSpec(
        "mlir-pass-cl-opt",
        False,
        "Do not declare global `cl::opt`/`cl::list` options in MLIR pass files; define TableGen pass options instead.",
    ),
    CheckSpec(
        "mlir-explicit-inline",
        False,
        "Omit `inline` on `static` functions in `.cpp` files (`static` alone lets the compiler inline).",
    ),
    CheckSpec(
        "mlir-camelback-naming",
        False,
        "Use `camelBack` for variable names in MLIR C++ code (neither `UpperCamelCase` nor `snake_case`).",
    ),
    CheckSpec(
        "mlir-doxygen-comment",
        False,
        "Use `///` Doxygen comments (not plain `//`) for doc comments on functions, classes, and structs in MLIR.",
    ),
    CheckSpec(
        "mlir-auto-overuse",
        False,
        "Spell out the concrete MLIR type instead of `auto` when the type does not appear on the right-hand side.",
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
]

SPEC_BY_NAME: Dict[str, CheckSpec] = {spec.name: spec for spec in CHECK_SPECS}


def _in_ranges(lineno: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= lineno <= end for start, end in ranges)


def _is_camel_back(ident: str) -> bool:
    if not ident or ident.startswith("_"):
        return True
    if "_" in ident:
        return False
    return ident[0].islower()


def check_cpp_source(path: str, raw: str) -> List[Finding]:
    findings: List[Finding] = []
    raw_lines = raw.splitlines()
    clean = strip_noise(raw)
    clean_lines = clean.splitlines()

    rewrite_ranges = find_bodied_blocks(clean, MATCH_AND_REWRITE_HEADER_RE)
    verify_ranges = find_bodied_blocks(clean, VERIFY_HEADER_RE)
    run_on_op_ranges = find_bodied_blocks(clean, RUN_ON_OP_HEADER_RE)

    is_cpp_impl = path.endswith((".cpp", ".cc"))
    norm_path = path.replace("\\", "/")
    is_pass_file = "/Passes/" in norm_path or norm_path.startswith("Passes/")

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

    # Track brace depth per line so doc-comment check only inspects namespace/class
    # scope, never inline comments inside function bodies.
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
        lineno = idx + 1

        if CONST_IR_PTR_RE.search(line):
            add(idx, "mlir-const-ir-pointer", SPEC_BY_NAME["mlir-const-ir-pointer"].description)

        if CONST_IR_VALUE_RE.search(line):
            add(idx, "mlir-const-ir-value", SPEC_BY_NAME["mlir-const-ir-value"].description)

        if PASS_IR_BY_CONST_REF_RE.search(line) and not COMPARISON_SIG_RE.search(line):
            # Only flag function parameter lists (lines with `(` or `,` context, not local refs).
            if "(" in line or (idx > 0 and "(" in clean_lines[idx - 1]):
                add(idx, "mlir-pass-ir-by-value", SPEC_BY_NAME["mlir-pass-ir-by-value"].description)

        if BUILDER_BY_VALUE_PARAM_RE.search(line):
            add(idx, "mlir-builder-by-value", SPEC_BY_NAME["mlir-builder-by-value"].description)
        elif _in_ranges(lineno, rewrite_ranges) and RAW_OPBUILDER_IN_REWRITE_RE.search(line):
            add(idx, "mlir-raw-builder-in-rewrite", SPEC_BY_NAME["mlir-raw-builder-in-rewrite"].description)

        if _in_ranges(lineno, rewrite_ranges):
            if DIRECT_ERASE_OR_MOVE_RE.search(line):
                add(idx, "mlir-direct-ir-mutation", SPEC_BY_NAME["mlir-direct-ir-mutation"].description)
            else:
                rauw = DIRECT_RAUW_RE.search(line)
                if rauw and "rewriter" not in rauw.group(1).lower():
                    add(idx, "mlir-direct-ir-mutation", SPEC_BY_NAME["mlir-direct-ir-mutation"].description)

        if BUILDER_CREATE_RE.search(line):
            add(idx, "mlir-op-create", SPEC_BY_NAME["mlir-op-create"].description)

        if _in_ranges(lineno, verify_ranges) and NON_LOCAL_DEF_USE_RE.search(line):
            add(idx, "mlir-verifier-non-local", SPEC_BY_NAME["mlir-verifier-non-local"].description)

        if is_pass_file and PASS_CL_OPT_RE.search(line):
            add(idx, "mlir-pass-cl-opt", SPEC_BY_NAME["mlir-pass-cl-opt"].description)

        if _in_ranges(lineno, run_on_op_ranges) and PASS_ABORT_RE.search(line):
            add(idx, "mlir-pass-no-abort", SPEC_BY_NAME["mlir-pass-no-abort"].description)

        if BOOL_ERROR_RETURN_RE.search(line):
            add(idx, "mlir-bool-error-return", SPEC_BY_NAME["mlir-bool-error-return"].description)

        if is_cpp_impl and STATIC_INLINE_CPP_RE.search(line):
            add(idx, "mlir-explicit-inline", SPEC_BY_NAME["mlir-explicit-inline"].description)

        m_var = CAMELBACK_LOCAL_DECL_RE.match(line)
        if m_var:
            ident = m_var.group(1)
            if not _is_camel_back(ident):
                add(
                    idx,
                    "mlir-camelback-naming",
                    f"Variable `{ident}` should use `camelBack` naming in MLIR C++ code.",
                )

        if AUTO_NON_OBVIOUS_RE.search(line):
            add(idx, "mlir-auto-overuse", SPEC_BY_NAME["mlir-auto-overuse"].description)

        # Doxygen `///` check: non-Doxygen `//` comment at top-level or class scope
        # directly above a function/class/struct declaration.
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
            add(idx, "mlir-lit-stderr-devnull", SPEC_BY_NAME["mlir-lit-stderr-devnull"].description)

        if MLIR_CLOSING_BRACE_CHECK_RE.match(line):
            add(idx, "mlir-lit-closing-brace", SPEC_BY_NAME["mlir-lit-closing-brace"].description)

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
        git_cwd = first_root if os.path.isdir(first_root) else os.path.dirname(os.path.abspath(first_root))
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
