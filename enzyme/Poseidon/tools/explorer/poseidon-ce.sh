#!/usr/bin/env bash
# Compiler Explorer entry for Poseidon: clang's command line in, the rewritten
# program out. Between the two, the program is profiled: the source is built
# with -poseidon-profile-generate, run once (no arguments, stdin closed, under
# POSEIDON_CE_PROFILE_SECONDS and POSEIDON_CE_PROFILE_MEMORY_MB of address
# space, defaults 10 and 1024), and the profile feeds the compile Compiler
# Explorer asked for. That compile is the search: it gets
# POSEIDON_CE_SOLVE_SECONDS (default 60) and POSEIDON_CE_SOLVE_MEMORY_MB of
# address space per process (default 4096), and past either the program is
# compiled as written. At most POSEIDON_CE_MAX_SEARCHES compiles (default 2)
# profile and search at once; one that finds every slot taken for 3 seconds is
# compiled as written.
# Every -poseidon-<flag> on the command line goes to the pass; -poseidon-tau
# defaults to POSEIDON_CE_TAU. Flags that only shape the output (-emit-llvm,
# remarks, -fstack-usage, -mllvm -print-*, ...) reach that last compile only;
# -E and -fsyntax-only go straight to clang. Remarks are reported as notes.
# The compile's <func>.explain.json files are left next to its output for
# poseidon-explain-ce.sh, unless POSEIDON_CE_EXPLAIN=0.
#
# The cost model is cost_models/cm_x86-64_explorer.csv, measured on the machine
# that runs the explorer with calibrate/microbm_cpu.py, or
# POSEIDON_CE_COST_MODEL. There is no default model.
#
# Herbie results are reused across compiles: read from cache/ next to this
# script (shipped for the examples) and from POSEIDON_CE_CACHE, where new ones
# are added and the search slots live. The DP table is not: it is keyed by
# function name.
#
# Layout next to this script (POSEIDON_CE_ROOT): lib/Poseidon-<N>.so,
# lib/ClangEnzyme-<N>.so, lib/libposeidon_profile.a,
# lib/libposeidon_profile_host.a, include/poseidon/, cost_models/,
# herbie/bin/herbie, calibrate/, and optionally VERSION and cache/. The clang
# is POSEIDON_CE_CLANG.
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
PROFILE_MEMORY="${POSEIDON_CE_PROFILE_MEMORY_MB:-1024}"
BUDGET="${POSEIDON_CE_SOLVE_SECONDS:-60}"
MEMORY="${POSEIDON_CE_SOLVE_MEMORY_MB:-4096}"
SEARCHES="${POSEIDON_CE_MAX_SEARCHES:-2}"
SHARED="${POSEIDON_CE_CACHE:-${TMPDIR:-/tmp}/poseidon-ce-cache}"
RECOMPILE="Clear cache & recompile, at the bottom of the compiler pane"

args=() final=() pass=() src="" out="" emit="" tau_given=0 verbose=0 opt="" direct=0 ir=0 dump=0
while [ $# -gt 0 ]; do
  case "$1 ${2:-}" in
    -mllvm\ -poseidon-*) shift ;;
    -Xclang\ -emit-llvm) ir=1; final+=("$1" "$2"); shift 2; continue ;;
    -mllvm\ -print-*|-mllvm\ --print-*) dump=1; final+=("$1" "$2"); shift 2; continue ;;
  esac
  case "$1" in
    -poseidon-tau=*) tau_given=1; pass+=(-mllvm "$1") ;;
    -poseidon-print|-poseidon-show-table) verbose=1; pass+=(-mllvm "$1") ;;
    -poseidon-*) pass+=(-mllvm "$1") ;;
    -o) out="$2"; shift ;;
    -S|-c) emit="$1" ;;
    -E) direct=1; args+=("$1") ;;
    -fsyntax-only) direct=1; final+=("$1") ;;
    -emit-llvm) ir=1; final+=("$1") ;;
    -R*|-fsave-optimization-record*|-foptimization-record-*|-fstack-usage|-ftime-trace*|-fdiagnostics-show-hotness|-fdiagnostics-hotness-threshold=*) final+=("$1") ;;
    -O*) opt="$1"; args+=("$1") ;;
    *.c|*.cc|*.cpp|*.cxx|*.cu) src="$1" ;;
    *) args+=("$1") ;;
  esac
  shift
done
[ -n "$src" ] || { echo "poseidon-ce: no source file on the command line" >&2; exit 1; }
if [ "$direct" = 1 ] && [ "$ir" = 0 ]; then
  exec "$CLANG" -ffp-contract=on "-I$ROOT/include" "${args[@]}" "${final[@]}" ${emit:+"$emit"} "$src" ${out:+-o "$out"}
fi
if [ ! -f "$COST_MODEL" ]; then
  echo "poseidon-ce: error: no cost model at $COST_MODEL; measure this machine's with 'python3 $ROOT/calibrate/microbm_cpu.py --native-arch x86-64 --out $ROOT/cost_models/cm_x86-64_explorer.csv' (needs numpy and llvmlite) or point POSEIDON_CE_COST_MODEL at one" >&2
  exit 1
fi
mkdir -p "$SHARED" || { echo "poseidon-ce: error: cannot create $SHARED (POSEIDON_CE_CACHE) for the Herbie results and the search slots" >&2; exit 1; }
[ "$tau_given" = 1 ] || pass+=(-mllvm "-poseidon-tau=$TAU")
case "${opt:--O0}" in
  -O0) echo "poseidon-ce: warning: compile with -O1 or higher; at -O0 every load of a value reaches the solver as a separate input, so most rewrites are out of reach" >&2 ;;
esac
common=(-ffp-contract=on "-I$ROOT/include" "${args[@]}")
link=()
[ -n "$emit" ] || link=("$ROOT/lib/libposeidon_profile.a" -lm)

take_slot() {
  local k try
  for ((try = 0; try <= 6; try++)); do
    [ "$try" = 0 ] || sleep 0.5
    for ((k = 0; k < SEARCHES; k++)); do
      exec {slot}>>"$SHARED/search-slot-$k" || return 1
      flock -n "$slot" && return 0
      exec {slot}>&-
    done
  done
  return 1
}
if ! take_slot; then
  "$CLANG" "${common[@]}" "${final[@]}" ${emit:+"$emit"} "$src" "${link[@]}" -o "$out"
  echo "poseidon-ce: note: the explorer is busy; compiled as written, try again in a minute ($RECOMPILE)" >&2
  exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/cache"
for dir in "$ROOT/cache" "$SHARED"; do
  for stamp in "$dir"/cachedHerbieOutput_*.txt.input; do
    [ -e "$stamp" ] || continue
    entry="${stamp%.input}"
    if [ -e "$entry" ]; then cp "$entry" "$entry".* "$work/cache/" 2>/dev/null || true; fi
  done
done
publish() {
  exec {lock}>>"$SHARED/publish.lock" || return 0
  flock -w 10 "$lock" || { exec {lock}>&-; return 0; }
  for entry in "$work"/cache/cachedHerbieOutput_*.txt; do
    [ -e "$entry.input" ] || continue
    name="${entry##*/}"
    [ ! -e "$SHARED/$name" ] || continue
    ! grep -q '"status":"timeout"' "$entry" || continue
    for f in "$entry".* "$entry"; do
      if cp "$f" "$SHARED/.${f##*/}.$$"; then mv -f "$SHARED/.${f##*/}.$$" "$SHARED/${f##*/}" || true; fi
    done
  done
  exec {lock}>&-
}
diagnostics() {
  esc=$'\e'
  if [ "$dump" = 1 ]; then
    find "$work/ir" -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null | awk -F- '{ print $1 "\t" ($NF == "before.ll" ? 0 : 1) "\t" $0 }' | sort -k1,1n -k2,2n |
      cut -f3- | while IFS= read -r f; do cat "$work/ir/$f"; done
    sed -n '/^\(# \)\{0,1\}\*\*\* IR Dump/,$p' "$work/solve.err" | sed "s/$esc\\[[0-9;]*m//g" |
      grep -avE '^[^ ]+:[0-9]+:[0-9]+: |^ +[0-9]* \| |^[0-9]+ (warning|error)s? generated' || true
  elif [ "$verbose" = 1 ]; then cat "$work/solve.err"
  else
    grep -aE "^($esc\\[[0-9;]*m)*(\\[poseidon\\]|Failed to)|No solution found|Best achievable|Applying solution for|warning: |error: |note: |remark: " "$work/solve.err" |
      grep -vE 'Starting Floodfill|initial subgraphs|After splitting|canonical form hash|\] Finished|PT Function identified|relative error tolerance:|NONE lands on the Pareto|regime-split candidate|dropping (array )?candidate|array candidate per element|Herbie core .*: status=[a-z]+-(start|target)|Herbie cache key collision' || true
  fi | sed -E "s/^(.*:[0-9]+:[0-9]+: ($esc\\[[0-9;]*m)*)remark: /\\1note: /" >&2
}

"$CLANG" "${common[@]}" "${load[@]}" "-fpass-plugin=$ENZYME" \
  -mllvm -poseidon-profile-generate "$src" "$ROOT/lib/libposeidon_profile_host.a" \
  "$ROOT/lib/libposeidon_profile.a" -lm \
  -o "$work/profiled" 2>"$work/profile.err" || { cat "$work/profile.err" >&2; exit 1; }
rc=0
(cd "$work" && ulimit -v $((PROFILE_MEMORY * 1024)) && POSEIDON_PROFILE_DIR="$work/profile" timeout -k 2 "$SECS" ./profiled </dev/null >"$work/profile.out" 2>&1) {slot}>&- 2>/dev/null || rc=$?
if ! ls "$work"/profile/*.fpprofile >/dev/null 2>&1; then
  # A program whose static arrays alone exceed the cap is killed while it is
  # loaded, before it can print anything.
  image="$(size "$work/profiled" 2>/dev/null | awk 'NR == 2 { print int(($1 + $2 + $3) / 1048576) }')"
  if grep -qaE "^Poseidon FPProfiler: (out of memory|cannot allocate a)|instance of 'std::bad_alloc'|Cannot allocate memory|failed to map segment" "$work/profile.out" ||
    { [ "$rc" -gt 128 ] && [ "${image:-0}" -ge "$PROFILE_MEMORY" ]; }; then
    why="ran out of the ${PROFILE_MEMORY} MB of memory it gets here (the profiler adds a shadow copy of each array the site uses)"
  elif [ "$rc" = 124 ] || [ "$rc" = 137 ]; then why="did not finish within ${SECS}s"
  elif [ "$rc" -gt 128 ]; then why="was killed by SIG$(kill -l $((rc - 128)) 2>/dev/null || echo $((rc - 128)))"
  elif [ "$rc" != 0 ]; then why="exited with status $rc"
  fi
  if [ -n "${why:-}" ]; then
    echo "poseidon-ce: the profiling run (the program, run once with no arguments and no input) $why, so nothing was profiled; its last output:" >&2
    tail -20 "$work/profile.out" >&2
  else
    cat "$work/profile.err" >&2
    echo "poseidon-ce: no site was profiled; mark a function with POSEIDON_OPTIMIZE or a call with __poseidon_optimize" >&2
  fi
  exit 1
fi

[ "$dump" = 0 ] || final+=(-mllvm "-ir-dump-directory=$work/ir")
explain=0
[ "${POSEIDON_CE_EXPLAIN:-1}" = 0 ] || [ "$dump$ir" != 00 ] || { explain=1; final+=(-mllvm "-poseidon-explain=$work/explain"); }
solve=("$CLANG" "${common[@]}" "${final[@]}" "${load[@]}"
  -mllvm "-poseidon-profile-use=$work/profile" -mllvm "-poseidon-cost-model=$COST_MODEL"
  -mllvm "-poseidon-cache=$work/cache" -mllvm "-poseidon-herbie-binary=$HERBIE"
  -mllvm -poseidon-herbie-num-threads=2 -mllvm -poseidon-herbie-timeout=20
  -mllvm -poseidon-herbie-subgraph-timeout=60 -mllvm -poseidon-herbie-num-pts=256
  -mllvm -poseidon-herbie-num-iters=4 -mllvm -poseidon-herbie-num-enodes=4000
  -mllvm -poseidon-num-samples=256
  -mllvm -poseidon-print "${pass[@]}" ${emit:+"$emit"} "$src" "${link[@]}" -o "$out")
capped() { (ulimit -v $((MEMORY * 1024)) && exec "$@"); }
rc=0
capped timeout -k 5 "$BUDGET" "${solve[@]}" 2>"$work/solve.err" || rc=$?
publish
why=""
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
  why="Herbie did not finish within ${BUDGET}s"
  [ "$dump" = 1 ] || diagnostics
elif grep -qaE "^(out of memory|LLVM ERROR: out of memory|memory allocation of [0-9]+ bytes failed)|instance of 'std::bad_alloc'" "$work/solve.err"; then
  why="the search needed more than ${MEMORY} MB of memory"
fi
if [ -n "$why" ]; then
  rm -rf "$work/ir" "$work/explain" "$work/cache/table.json"
  echo "poseidon-ce: warning: $why, so this is the program as written. Searches that completed are kept; compiling again ($RECOMPILE) continues from them." >&2
  rc=0
  capped "${solve[@]}" -mllvm -poseidon-enable-herbie=0 -mllvm -poseidon-enable-pt=0 \
    -mllvm -poseidon-enable-multifloat=0 2>"$work/solve.err" || rc=$?
fi
exec {slot}>&-
[ "$rc" = 0 ] || { cat "$work/solve.err" >&2; exit 1; }
if [ "$explain" = 1 ]; then
  for f in "$work"/explain/*.explain.json; do
    [ -e "$f" ] || continue
    to="$(dirname "$out")/${f##*/}"
    { cp "$f" "$to.$$" && mv -f "$to.$$" "$to"; } || true
  done
fi
diagnostics
