#!/usr/bin/env python3
"""Flag direct LLVM API calls that have an Enzyme wrapper in Utils.h.

Enzyme builds against every LLVM release from 15 to main, so a handful of LLVM
APIs cannot be called directly: either their signature changed across that
range, or their contract did. Utils.h wraps each of those. Calling the LLVM API
directly compiles fine against whichever LLVM the author happened to have, and
breaks -- often only at runtime, under an assertion -- against the others.

Two such APIs are checked here.

`BasicBlock::getTerminator()` used as a null test
-------------------------------------------------

    if (BB->getTerminator())            // wrong
      B.SetInsertPoint(BB->getTerminator());

Through LLVM 22 `getTerminator()` returned null for a block that has none, so
this reads as "does this block have a terminator". As of LLVM 23 it asserts
instead:

    Assertion `hasTerminator() && "cannot get terminator of non-well-formed
    block"' failed.

so the test crashes on exactly the input it was written to handle. Use the
`hasTerminator()` helper from Utils.h, which is the version-correct spelling:

    if (hasTerminator(BB))              // right
      B.SetInsertPoint(BB->getTerminator());

Only *null tests* are flagged. `dyn_cast<BranchInst>(BB->getTerminator())` and
friends are fine -- those assume a terminator exists and test its type, which
is the normal assumption for a well-formed block.

`PointerType::get` / `PointerType::getUnqual`
---------------------------------------------

    PointerType::get(T, AS)             // typed pointers, gone in LLVM 17

The element-type overload was removed when typed pointers were. `Utils.h`
provides `getPointerType(T, AS)`, which picks the right spelling per version,
plus `getUnqual(T)` and `getInt8PtrTy(Ctx, AS)`. Call those instead of the LLVM
API.

Rebuilding a pointer type only to move it to another address space has its own
helper, `changePointerAddrSpace(PT, AS)`. Prefer it: reaching for
`getInt8PtrTy(Ctx, AS)` there happens to produce the right type, but it reads
as "make an i8*", so a genuine loss of the pointee type is indistinguishable
from a deliberate opaque pointer.

MLIR's unrelated `LLVM::LLVMPointerType::get` is not affected and not flagged.

`BasicBlock::getFirstNonPHI` and friends
----------------------------------------

    BB->getFirstNonPHI()                // removed on LLVM main

LLVM removed the instruction-returning `getFirstNonPHI` (llvm-project
62c5ede9fd14), leaving the iterator-flavoured `getFirstNonPHIIt` -- which in
turn does not exist before LLVM 18. `getFirstNonPHIOrDbg` changed its return
type to an iterator in LLVM 20 the same way. Utils.h wraps both; call the
free functions `getFirstNonPHI(BB)` / `getFirstNonPHIOrDbg(BB)` instead of
any member spelling.

Suppress a false positive with `// NOLINT(<check>)` on the flagged line, or
`// NOLINTNEXTLINE(<check>)` on the line before it, where `<check>` is
`terminator-null-test`, `enzyme-pointer-type` or `enzyme-first-non-phi`.
"""

import argparse
import os
import re
import sys

SOURCE_SUFFIXES = (".cpp", ".cc", ".h", ".hpp")
SKIP_DIRS = {".git", "build", "third_party", "external"}

# `X->getTerminator()` / `X.getTerminator()` with the result used as a boolean:
#   if (BB->getTerminator())            if (!BB->getTerminator())
#   if (auto T = BB->getTerminator())   BB->getTerminator() ? a : b
#   BB->getTerminator() == nullptr      nullptr != BB->getTerminator()
RECEIVER = r"[\w.>\-\[\]()*&:]*?"
TERMINATOR_NULL_TEST = re.compile(
    r"if\s*\(\s*!?\s*" + RECEIVER + r"getTerminator\s*\(\s*\)\s*\)"
    r"|if\s*\(\s*(?:auto|const\s+auto|[\w:]+\s*\*)\s*\*?\s*\w+\s*=\s*"
    + RECEIVER
    + r"getTerminator\s*\(\s*\)\s*\)"
    r"|getTerminator\s*\(\s*\)\s*(?:\?|[=!]=\s*(?:nullptr|NULL))"
    r"|(?:nullptr|NULL)\s*[=!]=\s*" + RECEIVER + r"getTerminator\s*\(\s*\)"
)

# PointerType::get(...) / PointerType::getUnqual(...), but not MLIR's
# LLVM::LLVMPointerType::get.
POINTER_TYPE = re.compile(
    r"(?<![\w:])(?:llvm::)?PointerType::(?:get|getUnqual)\s*\("
)

# Member-call spellings of the getFirstNonPHI family. The Utils.h wrappers are
# free functions, so `getFirstNonPHI(BB)` does not match: only `X->` / `X.`
# calls do. `getFirstNonPHIOrDbgOrLifetime` is not matched (the `\s*\(` must
# follow one of the listed suffixes immediately).
FIRST_NON_PHI = re.compile(r"(?:->|\.)\s*getFirstNonPHI(?:It|OrDbg)?\s*\(")

CHECKS = [
    (
        "terminator-null-test",
        TERMINATOR_NULL_TEST,
        "getTerminator() used as a null test; this returns null only through "
        "LLVM 22 and asserts from LLVM 23 on. Use hasTerminator(BB) from "
        "Utils.h.",
    ),
    (
        "enzyme-first-non-phi",
        FIRST_NON_PHI,
        "getFirstNonPHI/getFirstNonPHIIt/getFirstNonPHIOrDbg called as a "
        "member; no one spelling exists on every supported LLVM. Use the "
        "free functions getFirstNonPHI(BB) / getFirstNonPHIOrDbg(BB) from "
        "Utils.h.",
    ),
    (
        "enzyme-pointer-type",
        POINTER_TYPE,
        "PointerType::get/getUnqual called directly; the element-type overload "
        "does not exist on every supported LLVM. Use getPointerType/getUnqual/"
        "getInt8PtrTy from Utils.h, or changePointerAddrSpace() if you are "
        "only moving a pointer to another address space.",
    ),
]


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
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            for j in range(i, min(i + 2, n)):
                out[j] = " "
            i += 2
        elif c in "\"'":
            quote = c
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


def suppressed(raw_lines, idx, check):
    """True if line `idx` (0-based) carries a NOLINT for `check`."""
    same = re.compile(r"NOLINT\(\s*" + re.escape(check) + r"\s*\)")
    prev = re.compile(r"NOLINTNEXTLINE\(\s*" + re.escape(check) + r"\s*\)")
    if same.search(raw_lines[idx]):
        return True
    return idx > 0 and prev.search(raw_lines[idx - 1])


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
    args = parser.parse_args()

    files = list(iter_sources(args.roots))
    errors = 0

    for path in files:
        raw = open(path, encoding="utf-8", errors="replace").read()
        raw_lines = raw.splitlines()
        # Search the comment- and string-stripped text so that prose and
        # NOLINT markers cannot themselves trip a check, but report the
        # original line so the message shows the real code.
        lines = strip_noise(raw).splitlines()
        for check, pattern, msg in CHECKS:
            for idx, line in enumerate(lines):
                if not pattern.search(line):
                    continue
                if suppressed(raw_lines, idx, check):
                    continue
                errors += 1
                lineno = idx + 1
                print(f"{path}:{lineno}: error: {msg}\n    {raw_lines[idx].strip()}")
                if args.github:
                    print(
                        f"::error file={path},line={lineno},"
                        f"title=Direct LLVM API use ({check})::{msg}"
                    )

    if errors:
        print(
            f"\n{errors} direct use(s) of an LLVM API that Utils.h wraps. Use "
            f"the helper, or add `// NOLINT(<check>)` if the direct call is "
            f"provably correct on every supported LLVM.",
            file=sys.stderr,
        )
        return 1
    print(f"checked {len(files)} files: no direct uses of wrapped LLVM APIs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
