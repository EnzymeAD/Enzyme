#!/usr/bin/env python3
"""Flag IR emission whose order depends on C++ argument evaluation order.

Building an op directly inside the argument list of another op-creating call:

    Value cond = stablehlo::OrOp::create(
        rewriter, loc,
        stablehlo::CompareOp::create(rewriter, loc, n, one, LE),
        stablehlo::CompareOp::create(rewriter, loc, s, one, LE));

is a hazard when *two or more* arguments do it. C++ leaves the evaluation
order of function arguments unspecified, so which compare is inserted into the
block first is the compiler's choice. The `or`'s operands are fixed, but the
textual order of the two compares is not: the same source emits them one way
on macOS and the other on Linux, which makes generated IR -- and any CHECK
line pinning it -- build-dependent.

The fix is always the same: bind each op to a local first, so the source
fixes the order.

    Value nSmall = stablehlo::CompareOp::create(rewriter, loc, n, one, LE);
    Value sSmall = stablehlo::CompareOp::create(rewriter, loc, s, one, LE);
    Value cond = stablehlo::OrOp::create(rewriter, loc, nSmall, sSmall);

An "emitter" is a call that inserts an op: `X::create(...)`,
`builder.create<X>(...)`, `rewriter.replaceOpWithNewOp<X>(...)`, or a call to
a function or lambda in this tree whose own body does one of those (helpers
such as `makeI64Constant`). Lambda *literals* passed as arguments do not
count -- their body runs inside the callee, after argument evaluation.

Suppress a false positive with `// NOLINT(emission-order)` on any line of the
flagged expression.
"""

import argparse
import os
import re
import sys

# A call that inserts an op into the IR.
MLIR_EMITTER = r"\b\w+::create\s*(?=\()|\bcreate\s*<|\breplaceOpWithNewOp\s*<"
# LLVM's IRBuilder has the same hazard; opt in with --include-llvm-builder.
LLVM_EMITTER = r"\bCreate[A-Z]\w*\s*(?=\()"
EMITTER_RE = re.compile(MLIR_EMITTER)

# Statements/expressions that are not calls we should look into.
KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "do",
    "else",
    "and",
    "or",
    "not",
    "decltype",
    "static_assert",
    "assert",
    "new",
    "delete",
    "throw",
    "case",
    "template",
    "operator",
    "noexcept",
    "constexpr",
    "alignof",
}

LAMBDA_ASSIGN_RE = re.compile(
    r"\b(?:auto|const\s+auto)\s*&?\s*(\w+)\s*=\s*\[[^\]]*\]\s*"
    r"(?:\([^)]*\))?\s*(?:mutable\s*)?(?:->[^{;]*)?\{"
)
CALL_RE = re.compile(r"(?<![\w])(\w+(?:::\w+)*)\s*(?:<[^<>;{}]*>)?\s*\(")
LAMBDA_ARG_RE = re.compile(r"^\s*\[[^\]]*\]\s*(?:\(|\{|mutable|->)")
SUPPRESS_RE = re.compile(r"NOLINT\(\s*emission-order\s*\)")

SOURCE_SUFFIXES = (".cpp", ".cc", ".h", ".hpp")
SKIP_DIRS = {".git", "build", "third_party", "external"}


def strip_noise(src):
    """Blank out comments, strings and char literals, preserving offsets."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif c in "\"'":
            quote = c
            out[i] = " "
            i += 1
            while i < n and src[i] != quote:
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n and src[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def match_paren(src, open_idx):
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def match_brace(src, open_idx):
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_args(src, lo, hi):
    """Split [lo, hi) on top-level commas, ignoring () [] {} and <> nesting."""
    args, depth, angle, start = [], 0, 0, lo
    for i in range(lo, hi):
        c = src[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "<" and depth == 0:
            angle += 1
        elif c == ">" and depth == 0 and angle > 0:
            angle -= 1
        elif c == "," and depth == 0 and angle == 0:
            args.append((start, i))
            start = i + 1
    args.append((start, hi))
    return args


def collect_helpers(src):
    """Names of functions and lambdas in `src` whose body inserts ops."""
    names = set()
    for m in LAMBDA_ASSIGN_RE.finditer(src):
        open_idx = src.index("{", m.end() - 1)
        close = match_brace(src, open_idx)
        if close > 0 and EMITTER_RE.search(src[open_idx:close]):
            names.add(m.group(1))
    for m in re.finditer(r"(?<![\w.>])(\w+)\s*\(", src):
        name = m.group(1)
        if name in KEYWORDS or len(name) < 3:
            continue
        close = match_paren(src, m.end() - 1)
        if close < 0:
            continue
        tail = src[close + 1 : close + 40]
        if not re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?\{", tail):
            continue
        open_idx = close + 1 + tail.index("{")
        body_end = match_brace(src, open_idx)
        if body_end > 0 and EMITTER_RE.search(src[open_idx:body_end]):
            names.add(name)
    return names


def iter_sources(roots):
    for root in roots:
        if os.path.isfile(root):
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith("bazel-")
            ]
            for name in sorted(filenames):
                if name.endswith(SOURCE_SUFFIXES):
                    yield os.path.join(dirpath, name)


def check_file(path, helper_re):
    raw = open(path, encoding="utf-8", errors="replace").read()
    src = strip_noise(raw)

    def emits(text):
        if LAMBDA_ARG_RE.match(text):
            return False  # lambda literal: its body runs inside the callee
        return bool(EMITTER_RE.search(text)) or bool(
            helper_re and helper_re.search(text)
        )

    findings, reported_lines = [], set()
    for m in CALL_RE.finditer(src):
        if m.group(1).split("::")[-1] in KEYWORDS:
            continue
        open_idx = src.index("(", m.end() - 1)
        close = match_paren(src, open_idx)
        if close < 0:
            continue
        args = split_args(src, open_idx + 1, close)
        if len(args) < 2:
            continue
        emitting = [a for a in args if emits(src[a[0] : a[1]])]
        if len(emitting) < 2:
            continue
        line = src.count("\n", 0, m.start()) + 1
        end_line = src.count("\n", 0, close) + 1
        if SUPPRESS_RE.search(raw[m.start() : close + 1]):
            continue
        # only report the outermost expression on a given line
        if line in reported_lines:
            continue
        reported_lines.add(line)
        findings.append(
            (
                line,
                end_line,
                m.group(1),
                " ".join(raw[m.start() : close + 1].split())[:200],
            )
        )
    return findings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roots", nargs="+", help="directories or files to check")
    parser.add_argument(
        "--github",
        action="store_true",
        help="also emit GitHub Actions error annotations",
    )
    parser.add_argument(
        "--include-llvm-builder",
        action="store_true",
        help="also treat LLVM IRBuilder Create* calls as "
        "emitters (same hazard, separate cleanup)",
    )
    args = parser.parse_args()

    if args.include_llvm_builder:
        global EMITTER_RE
        EMITTER_RE = re.compile(MLIR_EMITTER + "|" + LLVM_EMITTER)

    files = list(iter_sources(args.roots))
    helpers = set()
    stripped = {}
    for path in files:
        stripped[path] = strip_noise(
            open(path, encoding="utf-8", errors="replace").read()
        )
        helpers |= collect_helpers(stripped[path])
    helpers -= KEYWORDS
    helper_re = (
        re.compile(
            r"(?<![\w.>])(?:" + "|".join(sorted(map(re.escape, helpers))) + r")\s*\("
        )
        if helpers
        else None
    )

    total = 0
    for path in files:
        for line, end_line, callee, snippet in check_file(path, helper_re):
            total += 1
            msg = (
                f"{callee}(...) builds two or more ops in one argument list; "
                f"evaluation order is unspecified, so the emitted IR order "
                f"is compiler-dependent. Bind each to a local first."
            )
            print(f"{path}:{line}: {msg}\n    {snippet}")
            if args.github:
                print(
                    f"::error file={path},line={line},endLine={end_line},"
                    f"title=Nondeterministic IR emission order::{msg}"
                )

    if total:
        print(
            f"\n{total} nondeterministic emission site(s). "
            f"Bind each op to a local before the enclosing call, or add "
            f"`// NOLINT(emission-order)` if the order provably cannot be "
            f"observed.",
            file=sys.stderr,
        )
        return 1
    print(f"checked {len(files)} files: no nondeterministic emission sites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
