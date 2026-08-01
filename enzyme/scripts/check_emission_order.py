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

The second check is the same problem from a different source: iterating a
container that orders by pointer or hash value rather than by insertion.

    DenseMap<LLVMFuncOp, SmallVector<CallOpInterface>> kernelLaunches;
    ...
    for (auto &launch : kernelLaunches)   // order depends on pointer values
      ... create ops ...

This one is worse than the argument-order case: DenseMap/DenseSet order can
differ between two runs of the same binary, not just between toolchains. The
fix is a MapVector/SetVector, or sorting before the loop. Creating ops in the
body is an error; merely collecting elements into a vector is reported as a
warning, since whether that order reaches the output takes a human to judge.

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

# Containers whose iteration order is a function of pointer or hash values
# rather than of insertion order.
UNORDERED_TYPE_RE = re.compile(
    r"(?<![\w:])(?:llvm::|std::)?"
    r"(?:SmallPtrSet|SmallPtrSetImpl|SmallDenseMap|SmallDenseSet|DenseMap|"
    r"DenseMapBase|DenseSet|StringMap|StringSet|unordered_map|unordered_set|"
    r"unordered_multimap|unordered_multiset)\s*<"
)
RANGE_FOR_RE = re.compile(r"\bfor\s*\(\s*[^;{}]*?\s:\s*([^)]*)\)")
# Anything in a loop body that turns iteration order into program output.
ORDER_SENSITIVE_RE = re.compile(
    r"\b(?:push_back|emplace_back|append|emitError|emitWarning|emitRemark)\s*\("
)


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


def match_angle(src, open_idx):
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "<":
            depth += 1
        elif src[i] == ">":
            depth -= 1
            if depth == 0:
                return i
        elif src[i] in ";{}":
            return -1
    return -1


def brace_spans(src):
    """(start, end) of every brace pair, innermost last for a given point."""
    stack, spans = [], []
    for i, c in enumerate(src):
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            spans.append((stack.pop(), i))
    return spans


def unordered_decls(src):
    """(name, pos, scope_start, scope_end) per unordered-container declaration.

    Names are scoped to the innermost enclosing braces so that a parameter such
    as `SmallPtrSetImpl<Block *> &block` in one function does not make every
    other `block` in the file look like a hash container.
    """
    spans = brace_spans(src)
    decls = []
    for m in UNORDERED_TYPE_RE.finditer(src):
        close = match_angle(src, m.end() - 1)
        if close < 0:
            continue
        decl = re.match(r"\s*(?:const\s*)?[&*]?\s*(\w+)", src[close + 1 :])
        if not decl:
            continue
        pos = m.start()
        scope = min(
            (s for s in spans if s[0] < close < s[1]),
            key=lambda s: s[1] - s[0],
            default=(0, len(src)),
        )
        decls.append((decl.group(1), pos, scope[0], scope[1]))
    return decls


def iterated_name(expr):
    """The identifier a range-for actually iterates over, if any.

    `xs`, `*xs`, `xs.keys()`, `llvm::reverse(xs)` -> "xs".
    """
    expr = expr.strip()
    call = re.fullmatch(r"[\w:]+\s*\(\s*([^()]*?)\s*\)", expr)
    if call:
        expr = call.group(1).strip()
    m = re.match(r"[*&]?\s*(\w+)\s*(?:\.\w+\s*\(\s*\)\s*)?$", expr)
    return m.group(1) if m else None


def check_container_iteration(path, helper_re):
    """Loops over an unordered container that turn its order into output."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    src = strip_noise(raw)
    decls = unordered_decls(src)
    if not decls:
        return []

    findings = []
    for m in RANGE_FOR_RE.finditer(src):
        name = iterated_name(m.group(1))
        if name is None:
            continue
        at = m.start()
        hit = next(
            (
                n
                for n, pos, lo, hi in decls
                if n == name and pos < at and lo <= at <= hi
            ),
            None,
        )
        if hit is None:
            continue
        # A braced body runs to its matching brace; a brace-less one is the
        # single statement that follows.
        rest = src[m.end() :]
        lead = len(rest) - len(rest.lstrip())
        if rest[lead : lead + 1] == "{":
            open_idx = m.end() + lead
            close = match_brace(src, open_idx)
            if close < 0:
                continue
        else:
            close = src.find(";", m.end())
            if close < 0:
                continue
        body = src[m.end() : close]
        emits = bool(EMITTER_RE.search(body)) or bool(
            helper_re and helper_re.search(body)
        )
        if not (emits or ORDER_SENSITIVE_RE.search(body)):
            continue
        if SUPPRESS_RE.search(raw[m.start() : close + 1]):
            continue
        line = src.count("\n", 0, m.start()) + 1
        end_line = src.count("\n", 0, close) + 1
        # Creating ops in the body puts the order straight into the IR. Merely
        # collecting into a vector may or may not reach the output, so it is
        # reported but does not fail the run.
        findings.append(
            (
                line,
                end_line,
                hit,
                " ".join(raw[m.start() : m.end()].split()),
                "error" if emits else "warning",
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

    errors, warnings = 0, 0

    def report(path, line, end_line, msg, snippet, severity="error"):
        nonlocal errors, warnings
        if severity == "error":
            errors += 1
        else:
            warnings += 1
        print(f"{path}:{line}: {severity}: {msg}\n    {snippet}")
        if args.github:
            print(
                f"::{severity} file={path},line={line},endLine={end_line},"
                f"title=Nondeterministic IR emission order::{msg}"
            )

    for path in files:
        for line, end_line, callee, snippet in check_file(path, helper_re):
            report(
                path,
                line,
                end_line,
                f"{callee}(...) builds two or more ops in one argument list; "
                f"evaluation order is unspecified, so the emitted IR order "
                f"is compiler-dependent. Bind each to a local first.",
                snippet,
            )
        for line, end_line, name, snippet, severity in check_container_iteration(
            path, helper_re
        ):
            what = (
                "creates ops while iterating"
                if severity == "error"
                else "collects the elements of"
            )
            report(
                path,
                line,
                end_line,
                f"loop {what} `{name}`, but that container orders by pointer "
                f"or hash value rather than by insertion, so its order can "
                f"differ between runs. Use MapVector/SetVector, or sort first.",
                snippet,
                severity,
            )

    if warnings:
        print(
            f"\n{warnings} loop(s) over an unordered container feed something "
            f"order-sensitive but do not create ops directly; review whether "
            f"the order reaches the output.",
            file=sys.stderr,
        )
    if errors:
        print(
            f"\n{errors} nondeterministic emission site(s). See the guidance in "
            f"each message, or add `// NOLINT(emission-order)` if the order "
            f"provably cannot be observed.",
            file=sys.stderr,
        )
        return 1
    print(f"checked {len(files)} files: no nondeterministic emission sites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
