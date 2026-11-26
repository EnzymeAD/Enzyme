#!/usr/bin/env python3
"""cmpprof.py [--ignore KEY]... [--sum-tol T] [--by-function] <dir> <dir>

Compares two profile directories: every field but the atomic sums exactly, the
sums by relative difference (with --sum-tol, at most T). --ignore drops header
lines KEY = ...; --by-function matches files without the source hash a static
function's profile name ends in. Exit status 1 on any mismatch.
"""
import os
import re
import sys

SUMS = ("SumSens", "SumGrad", "SumAbsGrad")


def parse(path, ignore=()):
    header, slots, cur = [], {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("\t"):
            k, v = line.strip().split(" = ", 1)
            slots[cur][k] = v
        elif line.strip().isdigit():
            cur = int(line)
            slots[cur] = {}
        elif line.split(" = ")[0] not in ignore:
            header.append(line)
    return header, slots


def profiles(d, by_function):
    out = {}
    for f in os.listdir(d):
        if f.endswith(".fpprofile"):
            key = re.sub(r"~[0-9a-f]{16}(?=\.fpprofile$)", "", f) if by_function else f
            out[key] = os.path.join(d, f)
    return out


def main(a, b, ignore=(), tol=None, by_function=False):
    pa, pb = profiles(a, by_function), profiles(b, by_function)
    fa, fb = sorted(pa), sorted(pb)
    bad = 0
    if fa != fb:
        print("file sets differ:", sorted(set(fa) ^ set(fb)))
        bad = 1
    worst = 0.0
    for f in sorted(set(fa) & set(fb)):
        ha, sa = parse(pa[f], ignore)
        hb, sb = parse(pb[f], ignore)
        diffs = []
        if ha != hb:
            diffs.append("header")
        if sorted(sa) != sorted(sb):
            diffs.append("slot set")
        rel = 0.0
        for i in set(sa) & set(sb):
            for k in set(sa[i]) | set(sb[i]):
                x, y = sa[i].get(k), sb[i].get(k)
                if k in SUMS and x is not None and y is not None:
                    fx, fy = float(x), float(y)
                    d = abs(fx - fy) / max(abs(fx), abs(fy), 1e-300)
                    rel = max(rel, d)
                elif x != y:
                    diffs.append("slot %d %s: %s vs %s" % (i, k, x, y))
        worst = max(worst, rel)
        print("%s: %d slots, exact fields %s, sums max rel diff %.3e"
              % (f, len(sa), "IDENTICAL" if not diffs else "DIFFER " +
                 "; ".join(diffs[:5]), rel))
        bad |= bool(diffs)
    print("overall: %s, sums max rel diff %.3e"
          % ("exact fields identical" if not bad else "MISMATCH", worst))
    if tol is not None and worst > tol:
        print("sums differ by more than %.1e" % tol)
        bad = 1
    return bad


if __name__ == "__main__":
    args, ignore, tol, by_function = sys.argv[1:], [], None, False
    while args and args[0] in ("--ignore", "--sum-tol", "--by-function"):
        if args[0] == "--by-function":
            by_function = True
            args = args[1:]
            continue
        if args[0] == "--ignore":
            ignore.append(args[1])
        else:
            tol = float(args[1])
        args = args[2:]
    sys.exit(main(args[0], args[1], tuple(ignore), tol, by_function))
