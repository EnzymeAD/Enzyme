#!/usr/bin/env python3
"""Compare two Poseidon profile directories.

Every field except the four atomic sums must match exactly (header lines,
MinRes, MaxRes, Exec, NumOperands, Operand ranges); the sums are compared by
relative difference, whose maximum is printed. Exit status 1 on any exact
mismatch or a missing/extra file, and with --sum-tol T on a sum whose relative
difference exceeds T. --ignore KEY drops header lines "KEY = ..." from the
comparison (SiteId numbers sites per translation unit, so it differs between
a program built as one unit and the same program built as several).
"""
import os
import sys

SUMS = ("SumValue", "SumSens", "SumGrad", "SumAbsGrad")


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


def main(a, b, ignore=(), tol=None):
    fa = sorted(f for f in os.listdir(a) if f.endswith(".fpprofile"))
    fb = sorted(f for f in os.listdir(b) if f.endswith(".fpprofile"))
    bad = 0
    if fa != fb:
        print("file sets differ:", sorted(set(fa) ^ set(fb)))
        bad = 1
    worst = 0.0
    for f in sorted(set(fa) & set(fb)):
        ha, sa = parse(os.path.join(a, f), ignore)
        hb, sb = parse(os.path.join(b, f), ignore)
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
    args, ignore, tol = sys.argv[1:], [], None
    while args and args[0] in ("--ignore", "--sum-tol"):
        if args[0] == "--ignore":
            ignore.append(args[1])
        else:
            tol = float(args[1])
        args = args[2:]
    sys.exit(main(args[0], args[1], tuple(ignore), tol))
