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


def infer_activity(fsum, local):
    """Compose the per-function floating-point effects over the call graph
    (to a fixpoint; recursion is fine). Returns per function: per argument
    read/write/escape, globals read/written, unknown effects, and for each
    function that is active (writes floating-point data, returns a float or
    has unknown effects) the first reason found."""
    act = {}
    for n, f in list(fsum.items()) + [(k[1], v) for k, v in local.items()]:
        fa = f["activity"]
        act[n] = {"args": [{"r": x["read"], "w": x["write"], "e": x["escape"]}
                           for x in fa["args"]],
                  "gr": set(fa["globals_read"]), "gw": set(fa["globals_write"]),
                  "unknown": fa["unknown"], "returns_fp": fa["returns_fp"],
                  "frees": fa["frees"], "edges": fa["edges"], "ar": False,
                  "calls": f["calls"]}
    no_ir = set()
    changed = True
    while changed:
        changed = False
        for n, a in act.items():
            def setf(d, k, v=True):
                nonlocal changed
                if v and not d[k]:
                    d[k] = True
                    changed = True
            for root, callee, k in a["edges"]:
                c = act.get(callee)
                if c is None:
                    # no IR: assume it reads and writes what it is given
                    no_ir.add(callee)
                    r = w = True
                    e = False
                elif k < len(c["args"]):
                    r, w, e = c["args"][k]["r"], c["args"][k]["w"], c["args"][k]["e"]
                else:
                    r = w = e = False
                if root[0] == "a":
                    x = a["args"][int(root[1:])]
                    setf(x, "r", r)
                    setf(x, "w", w)
                    setf(x, "e", e)
                else:
                    g = root[1:]
                    if r and g not in a["gr"]:
                        a["gr"].add(g); changed = True
                    if w and g not in a["gw"]:
                        a["gw"].add(g); changed = True
                if e:
                    setf(a, "unknown")
            for callee in a["calls"]:
                c = act.get(callee)
                if c is None:
                    continue
                if not c["gr"] <= a["gr"]:
                    a["gr"] |= c["gr"]; changed = True
                if not c["gw"] <= a["gw"]:
                    a["gw"] |= c["gw"]; changed = True
                setf(a, "unknown", c["unknown"])
                setf(a, "frees", c["frees"])
                if c["ar"]:
                    setf(a, "ar", a["returns_fp"])
            # A returned float may carry a derivative only if it can depend
            # on floating-point data from outside: an argument, a global, an
            # unknown effect, or a callee's active return.
            setf(a, "ar", a["returns_fp"] and (
                a["unknown"] or bool(a["gr"]) or any(x["r"] for x in a["args"])))
    why = {}
    for n, a in act.items():
        if a["unknown"]:
            why[n] = "unknown effects"
        elif a["ar"]:
            why[n] = "returns a float computed from outside data"
        elif a["gw"]:
            why[n] = "writes " + ",".join(sorted(a["gw"])[:3])
        elif any(x["w"] for x in a["args"]):
            why[n] = "writes argument " + ",".join(
                str(i) for i, x in enumerate(a["args"]) if x["w"])
    act["__no_ir__"] = no_ir
    return {k: v for k, v in act.items() if k != "__no_ir__"}, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite-masks", type=int, default=0, metavar="CAP",
                    help="also export variants that assume some arguments are "
                         "not overwritten after the call (_o<mask>), at most "
                         "CAP per function besides the all-overwritten one; "
                         "0 (default) exports only the all-overwritten variant")
    ap.add_argument("--invariant-globals", action="store_true",
                    help="list the globals nothing reachable from the "
                         "differentiated functions writes (invariant_globals.txt)")
    ap.add_argument("--inactive", choices=["registered", "inferred", "both"],
                    default="registered",
                    help="which inactive functions stop the walk and get "
                         "registered: the __enzyme_inactivefn registrations, "
                         "those inferred from the summaries, or both")
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

    act, why = infer_activity(fsum, local)
    inferred = {n for n, x in act.items() if n not in why}
    registered = regs["inactive"] | {n for n, f in fsum.items() if f["inactive"]}
    inactive = {"registered": registered, "inferred": inferred,
                "both": registered | inferred}[a.inactive]

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
    # Arguments kept (not overwritten after the call) per call edge, top
    # down from each __enzyme_* call: a local object passed and not written
    # again later in the caller, or an argument of the caller that is kept
    # in the caller's own variant. Globals stay overwritten (exported
    # derivatives assume any later call may write them).
    kept_masks = defaultdict(set)      # function -> set of masks (tuples)
    if a.overwrite_masks:
        def fact(m, f):
            x = local.get((m, f)) or fsum.get(f)
            return x["activity"] if x else None
        work = []
        for (mod, caller, fn, tok) in roots:
            m = mod if (mod, fn) in local else fdef.get(fn)
            if m is None:
                continue
            n = len((fact(m, fn) or {}).get("args", []))
            work.append((m, fn, tuple([True] * n)))   # top level: kept
        seen = set()
        while work:
            m, f, mask = work.pop()
            if (m, f, mask) in seen:
                continue
            seen.add((m, f, mask))
            fa = fact(m, f)
            if fa is None:
                continue
            for c in fa.get("calls_at", []):
                g = c["callee"]
                if g in inactive or g in custom or is_runtime(g):
                    continue
                gm = m if (m, g) in local else fdef.get(g)
                if gm is None:
                    continue
                gmask = []
                for arg in c["args"]:
                    if arg is None:
                        gmask.append(False)
                        continue
                    r = arg["root"]
                    keep = not arg["after"] and (
                        r == "l" or (r[0] == "a" and int(r[1:]) < len(mask)
                                     and mask[int(r[1:])]))
                    gmask.append(keep)
                gmask = tuple(gmask)
                if any(gmask) and g in cross and gm != m:
                    if gmask in kept_masks[g] or len(kept_masks[g]) < a.overwrite_masks:
                        kept_masks[g].add(gmask)
                    else:
                        gmask = tuple(False for _ in gmask)
                work.append((gm, g, gmask))

        def hexmask(bits):
            out = ""
            for i in range(0, len(bits), 4):
                out += "0123456789abcdef"[sum(1 << j for j in range(4)
                                               if i + j < len(bits) and bits[i + j])]
            return out.rstrip("0")
        for m, e in exports.items():
            for g in e:
                base = e[g]
                e[g] = sorted(set(base) | {f"{t}+o{hexmask(k)}" for t in base
                                            for k in kept_masks.get(g, ())
                                            if hexmask(k)})
    with open(os.path.join(a.out, "import_variants.txt"), "w") as f:
        for m, e in sorted(exports.items()):
            for g in sorted(e):
                f.write(f"{g} {','.join(e[g])}\n")

    # Globals nothing reachable from the differentiated functions writes
    # (any data, not only floating-point): the reverse pass may read them
    # again instead of caching them.
    invariant = []
    if a.invariant_globals:
        reach, stack = set(), [(mod if (mod, fn) in local else fdef.get(fn), fn)
                                for (mod, _, fn, _) in roots]
        while stack:
            m, f = stack.pop()
            if m is None or (m, f) in reach:
                continue
            reach.add((m, f))
            for g in callees(m, f):
                if (m, g) in local:
                    stack.append((m, g))
                elif g in fdef:
                    stack.append((fdef[g], g))
        # arguments a function writes (any data), through its callees too
        def fx(m, f):
            return local.get((m, f)) or fsum.get(f)
        wany = {}
        for (m, f) in reach:
            wany[(m, f)] = list(fx(m, f)["activity"].get("args_write_any", []))
        changed = True
        while changed:
            changed = False
            for (m, f) in reach:
                for root, callee, k in fx(m, f)["activity"]["edges"]:
                    cm = m if (m, callee) in local else fdef.get(callee)
                    cw = wany.get((cm, callee))
                    w = True if cw is None else (k < len(cw) and cw[k])
                    if w and root[0] == "a":
                        i = int(root[1:])
                        if i < len(wany[(m, f)]) and not wany[(m, f)][i]:
                            wany[(m, f)][i] = True
                            changed = True
        written, unknown_any = set(), False
        for (m, f) in reach:
            x = local.get((m, f)) or fsum.get(f)
            fa = x["activity"]
            written |= set(fa.get("globals_write_any", []))
            # a global handed to a callee that writes that argument
            for root, callee, k in fa["edges"]:
                if root[0] == "g":
                    cm = m if (m, callee) in local else fdef.get(callee)
                    cw = wany.get((cm, callee))
                    if cw is None or (k < len(cw) and cw[k]):
                        written.add(root[1:])
            unknown_any |= fa.get("unknown_write", True) or fa["unknown"] \
                or x["indirect_calls"] > 0
            for c in x["calls"]:
                # a Fortran procedure without IR may write any COMMON block;
                # a C library function only what it is given (recorded as
                # the caller's own writes)
                if c not in fsum and not (m, c) in local and c.endswith("_") \
                        and not is_runtime(c):
                    unknown_any = True
        if not unknown_any:
            read = set()
            for (m, f) in reach:
                x = local.get((m, f)) or fsum.get(f)
                read |= set(x["globals"])
            allg = {g for s_ in mods.values() for g, i in s_["globals"].items()
                    if not i["constant"]}
            invariant = sorted((read & allg) - written)
    with open(os.path.join(a.out, "invariant_globals.txt"), "w") as f:
        for g in invariant:
            f.write(g + "\n")

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
    for g in sorted(inactive):
        r = escapes(g, set())
        (no_escape if r is False else escape if r else unknown).append(g)
    with open(os.path.join(a.out, "no_escape.c"), "w") as f:
        f.write("/* Generated by enzyme_thinlink.py: inactive routines that "
                "return no allocation to their callers. */\n")
        for g in no_escape:
            f.write(f"extern void {g}(void); __attribute__((used)) void "
                    f"*__enzyme_no_escaping_allocation_{g} = (void *){g};\n")

    # Registrations of the inactive functions this plan uses (replaces the
    # build's own list when --inactive=inferred).
    with open(os.path.join(a.out, "inactive.c"), "w") as f:
        f.write("/* Generated by enzyme_thinlink.py: inactive routines "
                f"({a.inactive}). */\n")
        for g in sorted(inactive):
            if g not in fsum and g not in regs["inactive"]:
                continue
            f.write(f"extern void {g}(void);\n")
            f.write(f"__attribute__((used)) void *__enzyme_inactivefn_{g} = "
                    f"(void *){g};\n")
            # the build's own registrations hold for routines without IR
            # (e.g. library routines), which the summaries cannot infer
            if (not act.get(g, {}).get("frees", True)
                    or g in regs.get("nofree", ())):
                f.write(f"__attribute__((used)) void *__enzyme_nofree_{g} = "
                        f"(void *){g};\n")
            if g in no_escape or g in regs.get("no_escape", ()):
                f.write(f"__attribute__((used)) void "
                        f"*__enzyme_no_escaping_allocation_{g} = (void *){g};\n")

    # Parameters of exported functions that never carry floating-point data:
    # no shadow, on both sides of every module boundary.
    with open(os.path.join(a.out, "inactive_params.txt"), "w") as f:
        nparams = 0
        for m, e in sorted(exports.items()):
            for g in sorted(e):
                if g not in act:
                    continue
                idx = [str(i) for i, x in enumerate(act[g]["args"])
                       if not (x["r"] or x["w"] or x["e"])]
                if idx:
                    f.write(f"{g} {','.join(idx)}\n")
                    nparams += len(idx)

    # The types declared on the parameters of exported functions, for the
    # modules that only declare them: type analysis would have seen them on
    # the definition with the whole program.
    with open(os.path.join(a.out, "param_types.txt"), "w") as f:
        for m, e in sorted(exports.items()):
            for g in sorted(e):
                for i, t in enumerate(fsum.get(g, {}).get("arg_types", [])):
                    if t:
                        f.write(f"{g}\t{i}\t{t}\n")

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
    if a.overwrite_masks:
        print(f"overwrite masks: {sum(len(v) for v in kept_masks.values())} kept-argument "
              f"variants for {len(kept_masks)} functions (cap {a.overwrite_masks})")
    if a.invariant_globals:
        print(f"invariant globals: {len(invariant)}")
    tot = sum(len(act[g]["args"]) for e in exports.values() for g in e if g in act)
    print(f"inactive parameters of exported functions: {nparams} of {tot}")
    reach = {f for (_, f) in need}
    print(f"inactive mode: {a.inactive}; registered {len(registered)}, "
          f"inferred {len(inferred)} (of {len(act)} functions with IR)")
    only_reg = sorted(registered - inferred)
    only_inf = sorted((inferred - registered) & (reach | {g for (m, f) in need
                      for g in callees(m, f)}))
    print(f"registered but inferred active ({len(only_reg)}):")
    for g in only_reg:
        print(f"    {g}: {why.get(g, 'no IR')}")
    print(f"inferred inactive, not registered, called from the closure "
          f"({len(only_inf)}): {only_inf}")
    plan["inferred_inactive"] = sorted(inferred)
    plan["registered_inactive_inferred_active"] = {g: why.get(g, "no IR")
                                                   for g in only_reg}
    json.dump(plan, open(os.path.join(a.out, "plan.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
