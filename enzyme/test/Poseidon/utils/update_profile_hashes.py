#!/usr/bin/env python3
"""Rewrite the CanonicalHash of the checked-in profile fixtures after a change
to the canonical form (an LLVM update, a change to Sites/Canonicalize.cpp).

A fixture under test/Poseidon/Inputs is a profile recorded once against the
canonical form of its site; its CanonicalHash names that form, and profile use
refuses a profile whose hash differs from the one the compile computes. When
the form changes, every test that reads the fixture stops with

  the profile of <site> was recorded against a different canonical form
  (profile CanonicalHash <old>, this compile <new>)

This script runs the suite of a build, reads those pairs from the failures,
and replaces <old> by <new> wherever the tree records it as a fixture's hash:
in each Inputs/**/*.fpprofile that carries it, and in the tests that copy a
fixture under a hash of their own (sed "s/^CanonicalHash = .*/.../") or check
it. A hash that no fixture carries, such as one a test writes to provoke the
error, is left alone. It repeats until a run reports no new pair, then lists
the tests that still fail.

Only the hash is rewritten: the records keep their numbers, which the tests
check. A record is kept for the slot index it was recorded under, so a form
change that also renumbers slots (an instruction added or removed before the
recorded ones) shows up as a test that still fails, and that fixture has to be
recorded again from its program.

  update_profile_hashes.py --build <build dir> [test path ...]

Test paths are relative to test/Poseidon (default: the whole suite except
Integration, which needs a GPU).
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
    for path in files(os.path.join(test_root, "Inputs"),
                      lambda n: n.endswith(".fpprofile")):
        hashes.update(FIXTURE_HASH.findall(open(path).read()))
    sed = re.compile(r"CanonicalHash = \.\*/CanonicalHash = ([0-9a-f]{16})/")
    for path in files(test_root, lambda n: n.endswith(TEST_SUFFIXES)):
        hashes.update(sed.findall(open(path).read()))
    return hashes


def run_lit(lit, build_root, tests):
    cmd = [sys.executable, lit, "-v", "--filter-out", "Integration/"]
    cmd += [os.path.join(build_root, t) for t in tests] or [build_root]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env).stdout
    failing = re.findall(r"^FAIL: Poseidon :: (\S+)", out, re.M)
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
