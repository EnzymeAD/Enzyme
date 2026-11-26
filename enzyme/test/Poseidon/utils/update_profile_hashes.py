#!/usr/bin/env python3
"""Rewrite the CanonicalHash of the checked-in profile fixtures after a change
to the canonical form (an LLVM update, a change to Sites/Canonicalize.cpp).

Runs the suite of a build, reads the (old, new) pairs of the "recorded against
a different canonical form" failures, and replaces old by new in each fixture
that carries it and in the tests that copy or check it, until a run reports no
new pair; then lists the tests that still fail. Only the hash is rewritten: a
form change that renumbers slots leaves a failing test, whose fixture has to
be recorded again from its program.

  update_profile_hashes.py --build <build dir> [test path ...]

Test paths are relative to test/Poseidon (default: all but Integration).
"""
import argparse
import os
import re
import subprocess
import sys

MISMATCH = re.compile(
    r"was recorded against a different canonical form "
    r"\(profile CanonicalHash ([0-9a-f]{16}), this compile ([0-9a-f]{16})\)")
FIXTURE_HASH = re.compile(r"^CanonicalHash = ([0-9a-f]{16})$", re.M)
TEST_SUFFIXES = (".ll", ".c", ".cpp", ".cu", ".hip")


def files(root, pred):
    for dirpath, _, names in os.walk(root):
        for name in names:
            if pred(name):
                yield os.path.join(dirpath, name)


def recorded_hashes(test_root):
    """The hashes the tree records for its fixtures: every fixture's own, and
    every hash a test writes into a copy of one."""
    hashes = set()
    for path in files(test_root, lambda n: n.endswith(".fpprofile")):
        hashes.update(FIXTURE_HASH.findall(open(path).read()))
    sed = re.compile(r"CanonicalHash = \.\*/CanonicalHash = ([0-9a-f]{16})/")
    for path in files(test_root, lambda n: n.endswith(TEST_SUFFIXES)):
        hashes.update(sed.findall(open(path).read()))
    return hashes


def run_lit(lit, build_root, tests):
    cmd = [sys.executable, lit, "-v", "--filter-out", "Integration/"]
    cmd += [os.path.join(build_root, t) for t in tests] or [build_root]
    # A test that pipes a compile's errors into FileCheck reports them only in
    # FileCheck's input dump, which FileCheck otherwise leaves out when its
    # checks pass and the compile's exit status fails the test.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               FILECHECK_OPTS="-dump-input=always")
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env).stdout
    failing = re.findall(r"^FAIL: Poseidon :: (\S+)", out, re.M)
    # A test that sends a compile's errors to a file of its own leaves them in
    # its temporary files, which lit keeps after a failure.
    for test in failing:
        output = os.path.join(build_root, os.path.dirname(test), "Output")
        stem = os.path.basename(test) + ".tmp"
        if not os.path.isdir(output):
            continue
        for name in os.listdir(output):
            if not name.startswith(stem):
                continue
            path = os.path.join(output, name)
            paths = [path]
            if os.path.isdir(path):
                paths = files(path, lambda n: True)
            for p in paths:
                with open(p, errors="replace") as f:
                    out += f.read()
    return out, failing


def rewrite(test_root, renames):
    changed = []
    for path in files(test_root, lambda n: n.endswith(TEST_SUFFIXES +
                                                      (".fpprofile",))):
        text = open(path).read()
        new = text
        for old, fresh in renames.items():
            new = new.replace(old, fresh)
        if new != text:
            open(path, "w").write(new)
            changed.append(os.path.relpath(path, test_root))
    return changed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True,
                    help="the build directory the suite runs from")
    ap.add_argument("--lit", default=None,
                    help="llvm-lit (default: the one the build's CMake "
                         "cache names)")
    ap.add_argument("tests", nargs="*")
    args = ap.parse_args()

    test_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_root = os.path.join(args.build, "test", "Poseidon")
    lit = args.lit
    if not lit:
        for line in open(os.path.join(args.build, "CMakeCache.txt")):
            if line.startswith("LLVM_EXTERNAL_LIT:"):
                lit = line.split("=", 1)[1].strip()
    if not lit:
        sys.exit("give --lit: the build's CMake cache names no llvm-lit")

    subprocess.run(["cmake", "--build", args.build, "--target",
                    "poseidon-test-deps"], check=True)
    tests = args.tests
    for round_ in range(1, 6):
        known = recorded_hashes(test_root)
        out, failing = run_lit(lit, build_root, tests)
        renames, conflicts = {}, set()
        for old, fresh in MISMATCH.findall(out):
            if old not in known:
                continue
            if renames.get(old, fresh) != fresh:
                conflicts.add(old)
            renames[old] = fresh
        for old in conflicts:
            print("CanonicalHash %s is read by sites of two different forms; "
                  "left as it is" % old)
            del renames[old]
        if not renames:
            break
        for path in rewrite(test_root, renames):
            print("round %d: %s" % (round_, path))
        tests = failing
    if failing:
        print("still failing:")
        for t in failing:
            print("  " + t)
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
