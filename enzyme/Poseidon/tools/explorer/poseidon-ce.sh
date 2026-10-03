#!/usr/bin/env bash
# Compiler Explorer entry for Poseidon: clang's command line in, the rewritten
# program out. Between the two, the program is profiled: the source is built
# with -poseidon-profile-generate, run once (no arguments, stdin closed, under
# POSEIDON_CE_PROFILE_SECONDS), and the profile feeds the compile Compiler
# Explorer asked for. Every -poseidon-<flag> on the command line goes to the
# pass; -poseidon-tau defaults to POSEIDON_CE_TAU.
#
# Layout next to this script (POSEIDON_CE_ROOT): lib/Poseidon-<N>.so,
# lib/ClangEnzyme-<N>.so, lib/libposeidon_profile.a, include/poseidon/,
# cost_models/, herbie/bin/herbie. The clang is POSEIDON_CE_CLANG.
set -eu
ROOT="${POSEIDON_CE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
if [ -z "${POSEIDON_CE_CLANG:-}" ]; then
  for so in "$ROOT"/lib/Poseidon-*.so; do
    [ -e "$so" ] || continue
    v="${so##*/Poseidon-}"; v="${v%.so}"
    [ -x "/usr/bin/clang++-$v" ] && { POSEIDON_CE_CLANG="/usr/bin/clang++-$v"; break; }
  done
fi
CLANG="${POSEIDON_CE_CLANG:?no clang++ matching a lib/Poseidon-<N>.so}"
VER="$("$CLANG" --version | sed -n 's/.*clang version \([0-9]*\).*/\1/p' | head -1)"
# Anything that is not a compile of a source file (--version, -print-*, ...)
# is clang's business.
case " $* " in
  *.c\ *|*.cc\ *|*.cpp\ *|*.cxx\ *|*.cu\ *) ;;
  *) exec "$CLANG" "$@" ;;
esac
PLUGIN="$ROOT/lib/Poseidon-$VER.so"
ENZYME="$ROOT/lib/ClangEnzyme-$VER.so"
# Before LLVM 22 clang parses -mllvm before loading -fpass-plugin; -load
# registers the pass's flags first.
load=("-fpass-plugin=$PLUGIN")
[ "$VER" -ge 22 ] || load+=(-Xclang -load -Xclang "$PLUGIN")
TAU="${POSEIDON_CE_TAU:-1e-6}"
COST_MODEL="${POSEIDON_CE_COST_MODEL:-$ROOT/cost_models/cm_x86-64_explorer.csv}"
HERBIE="$ROOT/herbie/bin/herbie"
SECS="${POSEIDON_CE_PROFILE_SECONDS:-10}"

args=() pass=() src="" out="" emit="" tau_given=0 verbose=0
while [ $# -gt 0 ]; do
  case "$1" in
    -poseidon-tau=*) tau_given=1; pass+=(-mllvm "$1") ;;
    -poseidon-print|-poseidon-show-table) verbose=1; pass+=(-mllvm "$1") ;;
    -poseidon-*) pass+=(-mllvm "$1") ;;
    -o) out="$2"; shift ;;
    -S|-c) emit="$1" ;;
    *.c|*.cc|*.cpp|*.cxx|*.cu) src="$1" ;;
    *) args+=("$1") ;;
  esac
  shift
done
[ -n "$src" ] || { echo "poseidon-ce: no source file on the command line" >&2; exit 1; }
[ "$tau_given" = 1 ] || pass+=(-mllvm "-poseidon-tau=$TAU")

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
common=(-ffp-contract=on "-I$ROOT/include" "${args[@]}")
link=()
[ -n "$emit" ] || link=("$ROOT/lib/libposeidon_profile.a" -lm)

"$CLANG" "${common[@]}" "${load[@]}" "-fpass-plugin=$ENZYME" \
  -mllvm -poseidon-profile-generate "$src" "$ROOT/lib/libposeidon_profile.a" -lm \
  -o "$work/profiled" 2>"$work/profile.err" || { cat "$work/profile.err" >&2; exit 1; }
(cd "$work" && POSEIDON_PROFILE_DIR="$work/profile" timeout "$SECS" ./profiled </dev/null >"$work/profile.out" 2>&1) \
  || { echo "poseidon-ce: the profiling run failed or exceeded ${SECS}s" >&2; tail -20 "$work/profile.out" >&2; exit 1; }
ls "$work"/profile/*.fpprofile >/dev/null 2>&1 || { echo "poseidon-ce: no site was profiled; mark one with POSEIDON_OPTIMIZE or __poseidon_fp_optimize" >&2; exit 1; }

"$CLANG" "${common[@]}" "${load[@]}" \
  -mllvm "-poseidon-profile-use=$work/profile" -mllvm "-poseidon-cost-model=$COST_MODEL" \
  -mllvm "-poseidon-cache=$work/cache" -mllvm "-poseidon-herbie-binary=$HERBIE" \
  -mllvm -poseidon-herbie-num-threads=4 -mllvm -poseidon-herbie-timeout=20 \
  -mllvm -poseidon-herbie-subgraph-timeout=60 -mllvm -poseidon-num-samples=256 \
  -mllvm -poseidon-print "${pass[@]}" \
  ${emit:+"$emit"} "$src" "${link[@]}" -o "$out" 2>"$work/solve.err" \
  || { cat "$work/solve.err" >&2; exit 1; }
# What the solver did, as compiler diagnostics Compiler Explorer shows; the
# whole log on -poseidon-print.
if [ "$verbose" = 1 ]; then cat "$work/solve.err" >&2
else grep -E '^\[poseidon\]|No solution found|Best achievable|Applying solution|warning: Poseidon' "$work/solve.err" >&2 || true
fi
