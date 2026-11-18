#!/usr/bin/env python3
"""Generate a Herbie platform .rkt file from a Poseidon GPU cost model CSV.

Usage:
    python3 csv_to_herbie_platform.py \\
        --csv cm_sm_<cc>_<device>.csv \\
        --arch sm_<cc> \\
        --device <device-name> \\
        --output <same dir>/cm_sm_<cc>_<device>.herbie.rkt

    # Or dump to stdout:
    python3 csv_to_herbie_platform.py --csv ... --arch sm_<cc>

The file is passed to Herbie by path (`herbie report --platform <path>`), so it
lives next to the CSV it was generated from and no Herbie tree is patched.

WHY THE FILE IS A `(module ...)` FORM AND NOT `#lang s-exp ...`:
Herbie's `activate-platform!` `dynamic-require`s the path. The Herbie this
project builds and ships is a `raco exe` / `raco distribute` executable, which
carries no Racket collection directories: reading a `#lang s-exp "..."` module
needs the `s-exp/lang/reader` collection and fails with `collection not found`.
A plain `(module <name> <language> ...)` form needs no reader, and the language
is named by the module name `raco exe` gave the embedded
`src/syntax/platform-language.rkt`. Use --platform-language / --flonum-module
for a Herbie that is not a `raco exe` binary (e.g. `herbie/syntax/platform-language`
for a collection install, or `(file "/abs/path/src/syntax/platform-language.rkt")`
for a source checkout).

Caveats:
  - f32 and f64 only. Herbie has no binary16/bfloat16 representations,
    so half/bf16 rows in the CSV are silently dropped.
  - Any op missing from the CSV at the requested precision aborts the
    script with a hard error - no silent fallbacks. Add the row to the
    CSV (or remove the op from UNARY_MATH/BINARY_MATH/SPECIAL_UNARY).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional


def read_poseidon_csv(path: Path) -> Tuple[Dict[Tuple[str, str], float], Dict[str, str]]:
    costs: Dict[Tuple[str, str], float] = {}
    meta: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if "=" in body:
                key, val = body.split("=", 1)
                meta[key.strip()] = val.strip()
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        op, precision, raw_cost = parts
        try:
            costs[(op, precision)] = int(raw_cost)
        except ValueError:
            try:
                costs[(op, precision)] = float(raw_cost)
            except ValueError:
                sys.stderr.write(f"warn: skipping unparsable row {raw!r}\n")
    return costs, meta


UNARY_MATH: List[Tuple[str, str, str]] = [
    ("fabs",   "fabs",   "fabs"),
    ("sin",    "sin",    "sin"),
    ("cos",    "cos",    "cos"),
    ("tan",    "tan",    "tan"),
    ("sinh",   "sinh",   "sinh"),
    ("cosh",   "cosh",   "cosh"),
    ("tanh",   "tanh",   "tanh"),
    ("asin",   "asin",   "asin"),
    ("acos",   "acos",   "acos"),
    ("atan",   "atan",   "atan"),
    ("asinh",  "asinh",  "asinh"),
    ("acosh",  "acosh",  "acosh"),
    ("atanh",  "atanh",  "atanh"),
    ("cbrt",   "cbrt",   "cbrt"),
    ("ceil",   "ceil",   "ceil"),
    ("floor",  "floor",  "floor"),
    ("round",  "round",  "round"),
    ("rint",   "rint",   "rint"),
    ("trunc",  "trunc",  "trunc"),
    ("erf",    "erf",    "erf"),
    ("exp",    "exp",    "exp"),
    ("exp2",   "exp2",   "exp2"),
    ("lgamma", "lgamma", "lgamma"),
    ("log",    "log",    "log"),
    ("log10",  "log10",  "log10"),
    ("log2",   "log2",   "log2"),
    ("sqrt",   "sqrt",   "sqrt"),
    ("tgamma", "tgamma", "tgamma"),
]

BINARY_MATH: List[Tuple[str, str, str]] = [
    ("pow",       "pow",       "pow"),
    ("atan2",     "atan2",     "atan2"),
    ("copysign",  "copysign",  "copysign"),
    ("fdim",      "fdim",      "fdim"),
    ("fmax",      "maxnum",    "fmax"),
    ("fmin",      "minnum",    "fmin"),
    ("fmod",      "fmod",      "fmod"),
    ("remainder", "remainder", "remainder"),
]

SPECIAL_UNARY: List[Tuple[str, str, str, str, str]] = [
    ("erfc",  "erf",   "(- 1 (erf x))", "erfc",  "(erfc x)"),
    ("expm1", "expm1", "(- (exp x) 1)", "expm1", "(expm1 x)"),
    ("log1p", "log1p", "(log (+ 1 x))", "log1p", "(log1p x)"),
]


def cost(costs: Dict[Tuple[str, str], float],
         op: str,
         precision: str) -> float:
    if (op, precision) in costs:
        return costs[(op, precision)]
    raise RuntimeError(
        f"csv has no cost row for {op}@{precision}; "
        f"add it to the CSV or remove the op from this script's emit list"
    )


def _fmt(n) -> str:
    if isinstance(n, float) and n.is_integer():
        return str(int(n))
    return str(n)


PREC_CONFIG = {
    "float": dict(
        repr_name="<binary32>", herbie_prec="binary32", suffix="f32",
        bits=32, round_fn="flsingle", use_libm=True, libm_suffix="f"),
    "double": dict(
        repr_name="<binary64>", herbie_prec="binary64", suffix="f64",
        bits=64, round_fn=None, use_libm=True, libm_suffix=""),
}

CSV_TO_HERBIE = {"float": "binary32", "double": "binary64"}


def emit_precision_block(costs: Dict[Tuple[str, str], float],
                         precision: str) -> List[str]:
    cfg = PREC_CONFIG[precision]
    repr_name = cfg["repr_name"]
    herbie_prec = cfg["herbie_prec"]
    suffix = cfg["suffix"]
    bits = cfg["bits"]
    round_fn = cfg["round_fn"]
    use_libm = cfg["use_libm"]
    libm_suffix = cfg["libm_suffix"]
    move_cost_var = f"{bits}bit-move-cost"

    if round_fn:
        add_impl = f"(compose {round_fn} +)"
        sub_impl = f"(compose {round_fn} -)"
        mul_impl = f"(compose {round_fn} *)"
        div_impl = f"(compose {round_fn} /)"
        neg_impl = f"(compose {round_fn} -)"
        pi_impl = f"(const ({round_fn} pi))"
        e_impl = f"(const ({round_fn} (exp 1)))"
    else:
        add_impl, sub_impl, mul_impl, div_impl = "+", "-", "*", "/"
        neg_impl = "-"
        pi_impl = "(const pi)"
        e_impl = "(const (exp 1))"

    def _math_impl(herbie_name: str, libm_base: str) -> str:
        effective_suffix = libm_suffix if use_libm else "f"
        return f"(from-libm '{libm_base}{effective_suffix})"

    def c(op: str) -> str:
        return _fmt(cost(costs, op, precision))

    label = herbie_prec.upper()
    lines: List[str] = [
        f";;;;;;;;;;;;;;;;;;;;;;;;;;;;; {label} ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
        "",
        f"(define-representation {repr_name} #:cost {move_cost_var})",
        "",
        f"(define-operation (if.{suffix} [c <bool>] [t {repr_name}] [f {repr_name}]) {repr_name}",
        f"  #:spec (if c t f) #:impl if-impl",
        f"  #:cost (if-cost boolean-move-cost))",
        "",
    ]

    cmp_c = c("fcmp")
    lines += [
        f"(define-operations ([x {repr_name}] [y {repr_name}]) <bool>",
        f"  [==.{suffix} #:spec (== x y) #:impl =          #:cost {cmp_c}]",
        f"  [!=.{suffix} #:spec (!= x y) #:impl (negate =) #:cost {cmp_c}]",
        f"  [<.{suffix}  #:spec (< x y)  #:impl <          #:cost {cmp_c}]",
        f"  [>.{suffix}  #:spec (> x y)  #:impl >          #:cost {cmp_c}]",
        f"  [<=.{suffix} #:spec (<= x y) #:impl <=         #:cost {cmp_c}]",
        f"  [>=.{suffix} #:spec (>= x y) #:impl >=         #:cost {cmp_c}])",
        "",
    ]

    lines += [
        f"(define-operations () {repr_name} #:fpcore (! :precision {herbie_prec} _)",
        f"  [PI.{suffix}       #:spec (PI)       #:impl {pi_impl} #:fpcore PI       #:cost {move_cost_var}]",
        f"  [E.{suffix}        #:spec (E)        #:impl {e_impl} #:fpcore E        #:cost {move_cost_var}]",
        f"  [INFINITY.{suffix} #:spec (INFINITY) #:impl (const +inf.0) #:fpcore INFINITY #:cost {move_cost_var}]",
        f"  [NAN.{suffix}      #:spec (NAN)      #:impl (const +nan.0) #:fpcore NAN      #:cost {move_cost_var}])",
        "",
    ]

    lines += [
        f"(define-operation (neg.{suffix} [x {repr_name}]) {repr_name}",
        f"  #:spec (neg x) #:impl {neg_impl}",
        f"  #:fpcore (! :precision {herbie_prec} (- x)) #:cost {c('fneg')})",
        "",
    ]

    lines += [
        f"(define-operations ([x {repr_name}] [y {repr_name}]) {repr_name} #:fpcore (! :precision {herbie_prec} _)",
        f"  [+.{suffix} #:spec (+ x y) #:impl {add_impl} #:cost {c('fadd')}]",
        f"  [-.{suffix} #:spec (- x y) #:impl {sub_impl} #:cost {c('fsub')}]",
        f"  [*.{suffix} #:spec (* x y) #:impl {mul_impl} #:cost {c('fmul')}]",
        f"  [/.{suffix} #:spec (/ x y) #:impl {div_impl} #:cost {c('fdiv')}])",
        "",
    ]

    unary_lines = [
        f"(define-operations ([x {repr_name}]) {repr_name} #:fpcore (! :precision {herbie_prec} _)"
    ]
    for herbie_name, poseidon_op, libm_base in UNARY_MATH:
        c_val = _fmt(cost(costs, poseidon_op, precision))
        impl = _math_impl(herbie_name, libm_base)
        unary_lines.append(
            f"  [{herbie_name}.{suffix} #:spec ({herbie_name} x) "
            f"#:impl {impl} #:cost {c_val}]"
        )
    unary_lines[-1] = unary_lines[-1][:-1] + "])"
    lines += unary_lines + [""]

    binary_lines = [
        f"(define-operations ([x {repr_name}] [y {repr_name}]) {repr_name} #:fpcore (! :precision {herbie_prec} _)"
    ]
    for herbie_name, poseidon_op, libm_base in BINARY_MATH:
        c_val = _fmt(cost(costs, poseidon_op, precision))
        impl = _math_impl(herbie_name, libm_base)
        binary_lines.append(
            f"  [{herbie_name}.{suffix} #:spec ({herbie_name} x y) "
            f"#:impl {impl} #:cost {c_val}]"
        )
    binary_lines[-1] = binary_lines[-1][:-1] + "])"
    lines += binary_lines + [""]

    special_lines = [
        f"(define-operations ([x {repr_name}]) {repr_name} #:fpcore (! :precision {herbie_prec} _)"
    ]
    for herbie_name, poseidon_op, spec, libm_base, fpcore_form in SPECIAL_UNARY:
        c_val = _fmt(cost(costs, poseidon_op, precision))
        impl = _math_impl(herbie_name, libm_base)
        special_lines.append(
            f"  [{herbie_name}.{suffix} #:spec {spec} "
            f"#:impl {impl} "
            f"#:fpcore {fpcore_form} #:cost {c_val}]"
        )
    special_lines[-1] = special_lines[-1][:-1] + "])"
    lines += special_lines + [""]

    hypot_impl = _math_impl("hypot", "hypot")
    lines += [
        f"(define-operation (hypot.{suffix} [x {repr_name}] [y {repr_name}]) {repr_name}",
        f"  #:spec (sqrt (+ (* x x) (* y y))) #:impl {hypot_impl}",
        f"  #:fpcore (! :precision {herbie_prec} (hypot x y)) #:cost {c('hypot')})",
        "",
    ]

    fma_impl = _math_impl("fma", "fma")
    lines += [
        f"(define-operation (fma.{suffix} [x {repr_name}] [y {repr_name}] [z {repr_name}]) {repr_name}",
        f"  #:spec (+ (* x y) z) #:impl {fma_impl}",
        f"  #:fpcore (! :precision {herbie_prec} (fma x y z)) #:cost {c('fma')})",
        "",
    ]

    return lines


def emit_cast_operations(costs: Dict[Tuple[str, str], float],
                         precisions: List[str]) -> List[str]:
    prec_order = {"float": 0, "double": 1}

    lines = [
        ";;;;;;;;;;;;;;;;;;;;;;;;;;;;; CASTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
        "",
    ]

    for src_csv in precisions:
        for dst_csv in precisions:
            if src_csv == dst_csv:
                continue
            src_cfg = PREC_CONFIG[src_csv]
            dst_cfg = PREC_CONFIG[dst_csv]
            src_repr = src_cfg["repr_name"]
            dst_repr = dst_cfg["repr_name"]
            src_herbie = src_cfg["herbie_prec"]
            dst_herbie = dst_cfg["herbie_prec"]
            op_name = f"{src_herbie}->{dst_herbie}"

            widening = prec_order[src_csv] < prec_order[dst_csv]
            if widening:
                impl = "identity"
            else:
                impl = dst_cfg["round_fn"] or "identity"

            csv_key = (f"fpext_{src_csv}_to_{dst_csv}" if widening
                       else f"fptrunc_{src_csv}_to_{dst_csv}")
            cast_cost = _fmt(cost(costs, csv_key, src_csv))

            lines += [
                f"(define-operation ({op_name} [x {src_repr}]) {dst_repr}",
                f"  #:spec x #:fpcore (! :precision {dst_herbie} (cast x))"
                f" #:impl {impl} #:cost {cast_cost})",
                "",
            ]

    return lines


# The module the platform file is written in, and the module `flsingle` comes
# from. Both default to the names `raco exe` gives the modules it embedded,
# because the Herbie this project builds and ships is such an executable and
# carries no collection directories to resolve ordinary module paths against.
DEFAULT_PLATFORM_LANGUAGE = "'|#%embedded:syntax/platform-language:|"
DEFAULT_FLONUM_MODULE = "'|#%embedded:math/flonum:|"


def embedded_module_names(herbie_binary: Path) -> Tuple[Optional[str], Optional[str]]:
    """The names raco exe gave platform-language.rkt and math/flonum inside this
    binary: `syntax/platform-language` when built from the source tree,
    `herbie/syntax/platform-language` when Herbie was installed as a package."""
    data = herbie_binary.read_bytes()
    import re
    lang = re.search(rb"#%embedded:([A-Za-z0-9_./-]*/)?syntax/platform-language:", data)
    flonum = re.search(rb"#%embedded:math/flonum:", data)
    return (f"'|{lang.group(0).decode()}|" if lang else None,
            f"'|{flonum.group(0).decode()}|" if flonum else None)


def emit_platform(costs: Dict[Tuple[str, str], float],
                  meta: Dict[str, str],
                  arch: str,
                  device: Optional[str],
                  platform_name: str,
                  language: str,
                  flonum_module: str) -> str:
    csv_precs = set(prec for (_, prec) in costs.keys())
    active_precisions = [p for p in ("float", "double") if p in csv_precs]

    fadd_f32 = cost(costs, "fadd", "float")
    fadd_f64 = cost(costs, "fadd", "double")
    del fadd_f32, fadd_f64

    move_costs = {}
    for prec in active_precisions:
        bits = PREC_CONFIG[prec]["bits"]
        if bits not in move_costs:
            move_costs[bits] = _fmt(cost(costs, "fcmp", prec))
    move_bool = move_costs.get(16, move_costs.get(32, "5"))

    header = [
        f";; CUDA {arch} platform, generated from a Poseidon GPU cost model CSV.",
    ]
    if device:
        header.append(f";; Target device: {device}")
    if "native_arch" in meta:
        header.append(f";; Source native_arch tag: {meta['native_arch']}")
    header += [
        ";; Cost unit: reciprocal throughput as the CSV measured it.",
        ";;",
        ";; Generated by poseidon/tools/herbie/csv_to_herbie_platform.py; pass it",
        ";; to Herbie by path (--platform <this file>). DO NOT HAND-EDIT:",
        ";; regenerate from the CSV instead (poseidon-calibrate --only",
        ";; herbie-platform --out <csv>).",
        "",
        f"(module {platform_name} {language}",
        "",
        f"(require {flonum_module})",
        "",
    ]
    for bits, mc in sorted(move_costs.items()):
        header.append(f"(define {bits}bit-move-cost   {mc})")
    header += [
        f"(define boolean-move-cost {move_bool})",
        "",
        ";;;;;;;;;;;;;;;;;;;;;;;;;;;;; BOOLEAN ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;",
        "",
        "(define-representation <bool> #:cost boolean-move-cost)",
        "",
        "(define-operations () <bool>",
        "  [TRUE  #:spec (TRUE)  #:impl (const true)  #:fpcore TRUE  #:cost boolean-move-cost]",
        "  [FALSE #:spec (FALSE) #:impl (const false) #:fpcore FALSE #:cost boolean-move-cost])",
        "",
        "(define-operations ([x <bool>] [y <bool>]) <bool>",
        "  [and #:spec (and x y) #:impl (lambda v (andmap values v)) #:cost boolean-move-cost]",
        "  [or  #:spec (or x y)  #:impl (lambda v (ormap values v))  #:cost boolean-move-cost])",
        "",
        "(define-operation (not [x <bool>]) <bool>",
        "  #:spec (not x) #:impl not #:cost boolean-move-cost)",
        "",
    ]

    body = header
    for prec in active_precisions:
        body += emit_precision_block(costs, prec)
    body += emit_cast_operations(costs, active_precisions)
    body += [")"]

    return "\n".join(body) + "\n"


def pct(s: str) -> str:
    """Escape a literal for an argparse help string."""
    return s.replace("%", "%%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path,
                    help="Poseidon cost model CSV (cm_<arch>_<device>.csv)")
    ap.add_argument("--arch", required=True,
                    help="Architecture tag as used in the CSV, e.g. sm_120")
    ap.add_argument("--platform-name",
                    help="Herbie platform name (default: 'cuda-<arch with underscores stripped>', "
                         "e.g. cuda-sm120)")
    ap.add_argument("--device",
                    help="Optional device name for the file-header comment (e.g. RTX5090)")
    ap.add_argument("--output", type=Path,
                    help="Output path for the .rkt file (default: stdout)")
    # argparse %-formats a help string against its own parameter dict, and the
    # two defaults contain a literal `%e` ("#%embedded:"), so every `%` in them
    # has to be doubled. Python 3.14 checks this when the argument is DECLARED,
    # not when --help is printed, so an unescaped default aborts every run.
    ap.add_argument("--platform-language", default=DEFAULT_PLATFORM_LANGUAGE,
                    help="Module path the platform is written in, used verbatim "
                         f"(default: {pct(DEFAULT_PLATFORM_LANGUAGE)}, the name raco exe "
                         "gives the embedded herbie src/syntax/platform-language.rkt)")
    ap.add_argument("--flonum-module", default=DEFAULT_FLONUM_MODULE,
                    help="Module path `flsingle` is required from, used verbatim "
                         f"(default: {pct(DEFAULT_FLONUM_MODULE)})")
    ap.add_argument("--herbie-binary", type=Path,
                    help="raco exe Herbie binary to read the two embedded module "
                         "names from; overrides the defaults when found")
    args = ap.parse_args()
    if args.herbie_binary and args.herbie_binary.exists():
        lang, flonum = embedded_module_names(args.herbie_binary)
        if lang:
            args.platform_language = lang
        if flonum:
            args.flonum_module = flonum
        print(f"platform language from {args.herbie_binary}: {args.platform_language}",
              file=sys.stderr)

    if not args.csv.exists():
        sys.exit(f"csv not found: {args.csv}")

    costs, meta = read_poseidon_csv(args.csv)
    if not any(op == "fadd" for (op, _) in costs):
        sys.exit(f"csv has no fadd entries - looks malformed: {args.csv}")

    have_float = any(prec == "float" for (_, prec) in costs)
    have_double = any(prec == "double" for (_, prec) in costs)
    if not have_float or not have_double:
        sys.exit(
            f"csv needs both float and double entries (found float={have_float}, "
            f"double={have_double}): {args.csv}"
        )
    csv_precs = sorted(set(prec for (_, prec) in costs.keys())
                       & {"float", "double"})
    print(f"precisions found in CSV: {', '.join(csv_precs)}", file=sys.stderr)

    platform_name = args.platform_name or f"cuda-{args.arch.replace('_', '')}"

    try:
        rkt = emit_platform(costs, meta, args.arch, args.device, platform_name,
                            args.platform_language, args.flonum_module)
    except RuntimeError as e:
        sys.exit(f"emit failed: {e}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rkt)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rkt)

    ops_per_prec = (
        len(UNARY_MATH) + len(BINARY_MATH) + len(SPECIAL_UNARY)
        + 4 + 6 + 4 + 1 + 1 + 1 + 1
    )
    n_precs = len(csv_precs)
    n_casts = n_precs * (n_precs - 1)
    total = ops_per_prec * n_precs + n_casts
    print(
        f"{platform_name}: ~{total} impls across {', '.join(csv_precs)} "
        f"({n_casts} casts, fadd f32={cost(costs, 'fadd', 'float')}, "
        f"fadd f64={cost(costs, 'fadd', 'double')})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
