#!/usr/bin/env bash
# Compiler Explorer entry for Poseidon: clang's command line in, the rewritten
# program out. Between the two, the program is profiled: the source is built
# with -poseidon-profile-generate, run once (no arguments, stdin closed, under
# POSEIDON_CE_PROFILE_SECONDS), and the profile feeds the compile Compiler
# Explorer asked for, which must finish within POSEIDON_CE_SOLVE_SECONDS.
# Every -poseidon-<flag> on the command line goes to the pass; -poseidon-tau
# defaults to POSEIDON_CE_TAU.
#
# Herbie results are reused across compiles: read from cache/ next to this
# script (shipped for the examples) and from POSEIDON_CE_CACHE, where new ones
# are added. The DP table is not: it is keyed by function name.
#
# Layout next to this script (POSEIDON_CE_ROOT): lib/Poseidon-<N>.so,
# lib/ClangEnzyme-<N>.so, lib/libposeidon_profile.a, include/poseidon/,
# cost_models/, herbie/bin/herbie, and optionally VERSION and cache/. The
# clang is POSEIDON_CE_CLANG.
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
if [ "$*" = --version ]; then
  "$CLANG" --version
  [ ! -f "$ROOT/VERSION" ] || echo "Poseidon $(cat "$ROOT/VERSION")"
  exit 0
fi
VER="$("$CLANG" --version | sed -n 's/.*clang version \([0-9]*\).*/\1/p' | head -1)"
# Anything that is not a compile of a source file (-print-*, --help, ...) is
# clang's business.
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
BUDGET="${POSEIDON_CE_SOLVE_SECONDS:-240}"
SHARED="${POSEIDON_CE_CACHE:-${TMPDIR:-/tmp}/poseidon-ce-cache}"

args=() pass=() src="" out="" emit="" tau_given=0 verbose=0 opt=""
while [ $# -gt 0 ]; do
  case "$1 ${2:-}" in -mllvm\ -poseidon-*) shift ;; esac
  case "$1" in
    -poseidon-tau=*) tau_given=1; pass+=(-mllvm "$1") ;;
    -poseidon-print|-poseidon-show-table) verbose=1; pass+=(-mllvm "$1") ;;
    -poseidon-*) pass+=(-mllvm "$1") ;;
    -o) out="$2"; shift ;;
    -S|-c) emit="$1" ;;
    -O*) opt="$1"; args+=("$1") ;;
    *.c|*.cc|*.cpp|*.cxx|*.cu) src="$1" ;;
    *) args+=("$1") ;;
  esac
  shift
done
[ -n "$src" ] || { echo "poseidon-ce: no source file on the command line" >&2; exit 1; }
[ "$tau_given" = 1 ] || pass+=(-mllvm "-poseidon-tau=$TAU")
case "${opt:--O0}" in
  -O0) echo "poseidon-ce: warning: compile with -O1 or higher; at -O0 every load of a value reaches the solver as a separate input, so most rewrites are out of reach" >&2 ;;
esac

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/cache"
for dir in "$ROOT/cache" "$SHARED"; do
  for stamp in "$dir"/cachedHerbieOutput_*.txt.input; do
    [ -e "$stamp" ] || continue
    entry="${stamp%.input}"
    [ -e "$entry" ] && cp "$entry" "$entry".* "$work/cache/" 2>/dev/null || true
  done
done
publish() {
  mkdir -p "$SHARED" 2>/dev/null || return 0
  for entry in "$work"/cache/cachedHerbieOutput_*.txt; do
    [ -e "$entry.input" ] || continue
    name="${entry##*/}"
    [ ! -e "$SHARED/$name" ] || continue
    ! grep -q '"status":"timeout"' "$entry" || continue
    for f in "$entry".* "$entry"; do
      cp "$f" "$SHARED/.${f##*/}.$$" && mv -f "$SHARED/.${f##*/}.$$" "$SHARED/${f##*/}" || true
    done
  done
}
diagnostics() {
  if [ "$verbose" = 1 ]; then cat "$work/solve.err" >&2
  else
    esc=$'\e'
    grep -aE "^($esc\\[[0-9;]*m)*(\\[poseidon\\]|Failed to)|No solution found|Best achievable|Applying solution|warning: |error: " "$work/solve.err" |
      grep -vE 'Starting Floodfill|initial subgraphs|After splitting|canonical form hash|\] Finished|PT Function identified|with relative error tolerance: 0\.0|accuracy target confidence|NONE lands on the Pareto' >&2 || true
  fi
}

common=(-ffp-contract=on "-I$ROOT/include" "${args[@]}")
link=()
[ -n "$emit" ] || link=("$ROOT/lib/libposeidon_profile.a" -lm)

"$CLANG" "${common[@]}" "${load[@]}" "-fpass-plugin=$ENZYME" \
  -mllvm -poseidon-profile-generate "$src" "$ROOT/lib/libposeidon_profile.a" -lm \
  -o "$work/profiled" 2>"$work/profile.err" || { cat "$work/profile.err" >&2; exit 1; }
rc=0
(cd "$work" && POSEIDON_PROFILE_DIR="$work/profile" timeout -k 2 "$SECS" ./profiled </dev/null >"$work/profile.out" 2>&1) 2>/dev/null || rc=$?
if ! ls "$work"/profile/*.fpprofile >/dev/null 2>&1; then
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then why="did not finish within ${SECS}s"
  elif [ "$rc" -gt 128 ]; then why="was killed by SIG$(kill -l $((rc - 128)) 2>/dev/null || echo $((rc - 128)))"
  elif [ "$rc" != 0 ]; then why="exited with status $rc"
  fi
  if [ -n "${why:-}" ]; then
    echo "poseidon-ce: the profiling run (the program, run once with no arguments and no input) $why, so nothing was profiled; its last output:" >&2
    tail -20 "$work/profile.out" >&2
  else
    cat "$work/profile.err" >&2
    echo "poseidon-ce: no site was profiled; on the host a site is a call wrapped in __poseidon_fp_optimize (POSEIDON_OPTIMIZE marks GPU kernels)" >&2
  fi
  exit 1
fi

solve=("$CLANG" "${common[@]}" "${load[@]}"
  -mllvm "-poseidon-profile-use=$work/profile" -mllvm "-poseidon-cost-model=$COST_MODEL"
  -mllvm "-poseidon-cache=$work/cache" -mllvm "-poseidon-herbie-binary=$HERBIE"
  -mllvm -poseidon-herbie-num-threads=4 -mllvm -poseidon-herbie-timeout=20
  -mllvm -poseidon-herbie-subgraph-timeout=60 -mllvm -poseidon-herbie-num-pts=256
  -mllvm -poseidon-herbie-num-iters=4 -mllvm -poseidon-herbie-num-enodes=4000
  -mllvm -poseidon-num-samples=256
  -mllvm -poseidon-print "${pass[@]}" ${emit:+"$emit"} "$src" "${link[@]}" -o "$out")
rc=0
timeout -k 5 "$BUDGET" "${solve[@]}" 2>"$work/solve.err" || rc=$?
publish
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
  diagnostics
  echo "poseidon-ce: warning: Herbie did not finish within ${BUDGET}s, so this is the program as written. Searches that completed are kept; compiling again continues from them." >&2
  rc=0
  "${solve[@]}" -mllvm -poseidon-enable-herbie=0 -mllvm -poseidon-enable-pt=0 \
    -mllvm -poseidon-enable-multifloat=0 2>"$work/solve.err" || rc=$?
fi
[ "$rc" = 0 ] || { cat "$work/solve.err" >&2; exit 1; }
diagnostics
