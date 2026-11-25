#!/usr/bin/env python3
"""Generate the Herbie platform of a Poseidon GPU cost model CSV.

    herbie_platform.py --csv cm_<arch>_<device>.csv --device <device> \\
        --herbie-binary <herbie> --output cm_<arch>_<device>.herbie.rkt

The platform is named after the CSV's native_arch: cuda-sm<NN> for an NVIDIA
GPU, hip-gfx<NNN> for an AMD one. Herbie loads it by path, and as a raco exe
binary carries no collection to read a #lang line with, the file is a plain
(module ...) form in the language module the binary embeds.

Herbie has binary32 and binary64 only, so the CSV's float and double rows are
read. A row the platform needs that the CSV lacks is an error.
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


# Herbie's operator, the CSV row and the libm function are one name.
UNARY_MATH: List[str] = [
    "fabs", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos",
    "atan", "asinh", "acosh", "atanh", "cbrt", "ceil", "floor", "round",
    "rint", "trunc", "erf", "exp", "exp2", "lgamma", "log", "log10", "log2",
    "sqrt", "tgamma",
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
        bits=32, round_fn="flsingle", libm_suffix="f"),
    "double": dict(
        repr_name="<binary64>", herbie_prec="binary64", suffix="f64",
        bits=64, round_fn=None, libm_suffix=""),
}

def emit_precision_block(costs: Dict[Tuple[str, str], float],
                         precision: str) -> List[str]:
    cfg = PREC_CONFIG[precision]
    repr_name = cfg["repr_name"]
    herbie_prec = cfg["herbie_prec"]
    suffix = cfg["suffix"]
    bits = cfg["bits"]
    round_fn = cfg["round_fn"]
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

    def _math_impl(libm_base: str) -> str:
        return f"(from-libm '{libm_base}{libm_suffix})"

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
    for herbie_name in UNARY_MATH:
        c_val = _fmt(cost(costs, herbie_name, precision))
        impl = _math_impl(herbie_name)
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
        impl = _math_impl(libm_base)
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
        impl = _math_impl(libm_base)
        special_lines.append(
            f"  [{herbie_name}.{suffix} #:spec {spec} "
            f"#:impl {impl} "
            f"#:fpcore {fpcore_form} #:cost {c_val}]"
        )
    special_lines[-1] = special_lines[-1][:-1] + "])"
    lines += special_lines + [""]

    hypot_impl = _math_impl("hypot")
    lines += [
        f"(define-operation (hypot.{suffix} [x {repr_name}] [y {repr_name}]) {repr_name}",
        f"  #:spec (sqrt (+ (* x x) (* y y))) #:impl {hypot_impl}",
        f"  #:fpcore (! :precision {herbie_prec} (hypot x y)) #:cost {c('hypot')})",
        "",
    ]

    fma_impl = _math_impl("fma")
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


def herbie_platform_name(arch: str) -> str:
    """cuda-sm120 for an NVIDIA architecture, hip-gfx942 for an AMD one: the
    name the pass gives Herbie (Target::herbiePlatform)."""
    vendor = "hip" if arch.startswith("gfx") else "cuda"
    return f"{vendor}-{arch.replace('_', '')}"


def emit_platform(costs: Dict[Tuple[str, str], float],
                  meta: Dict[str, str],
                  arch: str,
                  device: Optional[str],
                  platform_name: str,
                  language: str,
                  flonum_module: str) -> str:
    csv_precs = set(prec for (_, prec) in costs.keys())
    active_precisions = [p for p in ("float", "double") if p in csv_precs]

    move_costs = {}
    for prec in active_precisions:
        bits = PREC_CONFIG[prec]["bits"]
        if bits not in move_costs:
            move_costs[bits] = _fmt(cost(costs, "fcmp", prec))
    move_bool = move_costs[32]

    vendor = "HIP" if arch.startswith("gfx") else "CUDA"
    header = [
        f";; {vendor} {arch} platform, generated from a Poseidon GPU cost model CSV.",
    ]
    if device:
        header.append(f";; Target device: {device}")
    if "native_arch" in meta:
        header.append(f";; Source native_arch tag: {meta['native_arch']}")
    header += [
        ";; Cost unit: reciprocal throughput as the CSV measured it.",
        ";;",
        ";; Generated by herbie_platform.py; pass it",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path,
                    help="Poseidon cost model CSV (cm_<arch>_<device>.csv)")
    ap.add_argument("--device",
                    help="Device name for the file-header comment")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output path for the .rkt file")
    ap.add_argument("--herbie-binary", type=Path,
                    help="raco exe Herbie binary to read the two embedded module "
                         "names from; overrides the defaults when found")
    args = ap.parse_args()
    language, flonum_module = DEFAULT_PLATFORM_LANGUAGE, DEFAULT_FLONUM_MODULE
    if args.herbie_binary and args.herbie_binary.exists():
        lang, flonum = embedded_module_names(args.herbie_binary)
        language = lang or language
        flonum_module = flonum or flonum_module
        print(f"platform language from {args.herbie_binary}: {language}",
              file=sys.stderr)

    if not args.csv.exists():
        sys.exit(f"csv not found: {args.csv}")

    costs, meta = read_poseidon_csv(args.csv)
    arch = meta.get("native_arch")
    if not arch:
        sys.exit(f"csv has no '# native_arch=' header: {args.csv}")
    if not any(op == "fadd" for (op, _) in costs):
        sys.exit(f"csv has no fadd entries - looks malformed: {args.csv}")

    have_float = any(prec == "float" for (_, prec) in costs)
    have_double = any(prec == "double" for (_, prec) in costs)
    if not have_float or not have_double:
        sys.exit(
            f"csv needs both float and double entries (found float={have_float}, "
            f"double={have_double}): {args.csv}"
        )

    try:
        rkt = emit_platform(costs, meta, arch, args.device, herbie_platform_name(arch),
                            language, flonum_module)
    except RuntimeError as e:
        sys.exit(f"emit failed: {e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rkt)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
