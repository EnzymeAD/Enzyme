#!/usr/bin/env python3
"""Thin-link step for Enzyme separate compilation.

Reads the per-module facts written by `opt -passes=enzyme-summary` and plans
the per-module Enzyme step (opt -enzyme-separate-compilation):

  * which functions each module must export a derivative of, and in which
    variants (mode, strong zero, runtime activity, width): exactly those
    called, inside some differentiated call graph, from another module;
  * which modules must be compiled together (an __enzyme_* call whose
    differentiated function lives in another module: Enzyme needs its body);
  * which inactive callees return no allocation to their callers
    (__enzyme_no_escaping_allocation registrations);
  * the COMMON blocks that need shadows.

usage: enzyme_thinlink.py --out DIR SUMMARY.json...
Writes DIR/<module>.exports, DIR/no_escape.c, DIR/plan.json and prints a
report.

Typical flow (one summary per object, named <module>.json):
  ld.lld --thinlto-index-only=index.txt ... *.o      # the link's modules
  opt -load-pass-plugin=LLVMEnzyme-N.so -passes=enzyme-summary \
      -enzyme-summary-out=sum/<module>.json -disable-output <module>.o
  enzyme_thinlink.py --out plan sum/*.json
  opt ... -passes=preserve-nvvm,enzyme,preserve-nvvm-end \
      -enzyme-separate-compilation \
      -enzyme-export-list=plan/<module>.exports <module>.o   # per module
"""
import argparse
import json
import os
import sys
from collections import defaultdict

PREVAIL = {"external": 0, "weak": 1, "common": 2, "linkonce": 3,
           "available_externally": 9}


def variant_token(v):
    mode, sz, ra, width = v
    tok = mode
    if sz:
        tok += "+sz"
    if ra:
        tok += "+ra"
    if width != 1:
        tok += f"+w{width}"
    return tok


def is_runtime(name):
    # Callees Enzyme handles without a derivative from another module
    # (GradientUtils::usesExternalDerivative): flang runtime, Enzyme, LLVM.
    return name.startswith(("_Fortran", "__enzyme", "llvm."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("summaries", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    mods = {}
    for p in a.summaries:
        s = json.load(open(p))
        mod = os.path.basename(p)[: -len(".json")]
        mods[mod] = s

    # Prevailing definition of every function / global.
    fdef, fsum = {}, {}
    for mod, s in mods.items():
        for name, f in s["functions"].items():
            if f["linkage"] in ("internal", "private"):
                continue
            rank = PREVAIL.get(f["linkage"], 5)
            if name not in fdef or rank < PREVAIL.get(fsum[name]["linkage"], 5):
                fdef[name], fsum[name] = mod, f
    local = {(mod, n): f for mod, s in mods.items()
             for n, f in s["functions"].items()
             if f["linkage"] in ("internal", "private")}

    regs = defaultdict(set)
    for s in mods.values():
        for k, names in s["registrations"].items():
            regs[k] |= set(names)
    inactive = regs["inactive"] | {n for n, f in fsum.items() if f["inactive"]}
    custom = regs["custom_rule"]

    def callees(mod, name):
        f = local.get((mod, name)) or fsum.get(name)
        if f is None:
            return []
        return f["calls"] + [r for r in f["refs"] if r in fsum or (mod, r) in local]

    # Walk each differentiated call graph, carrying the variant.
    need = defaultdict(set)        # function -> variants its derivative is built under
    cross = defaultdict(set)       # function -> variants needed by other modules
    colocate, roots, missing = [], [], set()
    for mod, s in mods.items():
        for c in s["ad_calls"]:
            fn = c["fn"]
            v = (c["mode"], c["strong_zero"], c["runtime_activity"], c["width"])
            roots.append((mod, c["caller"], fn, variant_token(v)))
            if fn not in fdef and (mod, fn) not in local:
                missing.add(fn)
                continue
            if (mod, fn) not in local and fdef[fn] != mod:
                colocate.append((mod, fdef[fn], fn))
            stack = [(mod if (mod, fn) in local else fdef[fn], fn)]
            while stack:
                m, f = stack.pop()
                if v in need[(m, f)]:
                    continue
                need[(m, f)].add(v)
                for g in callees(m, f):
                    if g in inactive or g in custom or is_runtime(g):
                        continue
                    if (m, g) in local:
                        stack.append((m, g))
                    elif g in fdef:
                        if fdef[g] != m:
                            cross[g].add(v)
                        stack.append((fdef[g], g))
                    else:
                        missing.add(g)

    # Modules compiled together keep their calls local: no export needed.
    together = {}
    for m1, m2, _ in colocate:
        together.setdefault(m1, {m1}).add(m2)
        together[m2] = together[m1]
    exports = defaultdict(dict)
    for g, vs in cross.items():
        m = fdef[g]
        callers_elsewhere = True
        if m in together:
            # only keep if some caller module is outside the group
            callers_elsewhere = any(
                g in callees(cm, cf) and cm not in together[m]
                for (cm, cf) in need)
        if callers_elsewhere:
            exports[m][g] = sorted(variant_token(v) for v in vs)
    for m in mods:
        with open(os.path.join(a.out, f"{m}.exports"), "w") as f:
            for g in sorted(exports.get(m, {})):
                f.write(f"{g} {','.join(exports[m][g])}\n")

    # Non-escaping allocations: an inactive callee whose body (transitively,
    # through functions with IR) neither allocates nor returns a pointer.
    def escapes(name, seen):
        if name in seen:
            return False
        seen.add(name)
        f = fsum.get(name)
        if f is None:
            return None            # no IR: unknown
        if f["returns_pointer"] or f["allocates"]:
            return True
        return any(escapes(g, seen) for g in f["calls"] if g in fsum)
    no_escape, escape, unknown = [], [], []
    for g in sorted(regs["inactive"]):
        r = escapes(g, set())
        (no_escape if r is False else escape if r else unknown).append(g)
    with open(os.path.join(a.out, "no_escape.c"), "w") as f:
        f.write("/* Generated by enzyme_thinlink.py: inactive routines that "
                "return no allocation to their callers. */\n")
        for g in no_escape:
            f.write(f"extern void {g}(void); __attribute__((used)) void "
                    f"*__enzyme_no_escaping_allocation_{g} = (void *){g};\n")

    # COMMON blocks (shadowed through the build's shadow table).
    common = {}
    for s in mods.values():
        for g, info in s["globals"].items():
            if info["linkage"] == "common":
                common[g] = max(common.get(g, 0), info["size"])

    plan = {
        "roots": roots,
        "colocate": sorted(set(colocate)),
        "exports": {m: e for m, e in exports.items()},
        "no_escape": no_escape, "escaping_inactive": escape,
        "inactive_without_ir": unknown,
        "missing_callees": sorted(missing),
        "common_blocks": common,
        "closure_size": len({f for (_, f) in need}),
    }
    json.dump(plan, open(os.path.join(a.out, "plan.json"), "w"), indent=1)

    nexp = sum(len(e) for e in exports.values())
    nvar = sum(len(v) for e in exports.values() for v in e.values())
    print(f"modules: {len(mods)}, functions: {len(fsum)} (+{len(local)} local)")
    print(f"AD roots: {roots}")
    print(f"colocate: {plan['colocate']}")
    print(f"differentiated closure: {plan['closure_size']} functions")
    print(f"exports: {nexp} functions, {nvar} variants, from {len(exports)} modules")
    print(f"variants used: {sorted({t for e in exports.values() for v in e.values() for t in v})}")
    print(f"inactive: {len(regs['inactive'])} registered; no_escape {len(no_escape)}, "
          f"escaping {len(escape)} {escape[:8]}, no IR {len(unknown)} {unknown[:8]}")
    print(f"custom rules: {sorted(custom)}")
    print(f"callees without IR, not inactive/runtime: {sorted(missing)[:20]} ({len(missing)})")
    print(f"COMMON blocks: {len(common)}")


if __name__ == "__main__":
    sys.exit(main())
