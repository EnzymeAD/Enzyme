#!/usr/bin/env bash
# Lays out a build the way the nightly-poseidon tarball does:
#   package.sh <build-dir> <cost-model.csv|-> <out-dir>
# "-" ships no cost model: the machine that runs the explorer measures its own
# with calibrate/microbm_cpu.py. The examples are then compiled for their
# Herbie results with POSEIDON_CE_COST_MODEL, and without it none ship.
set -eu
[ $# -eq 3 ] || { echo "usage: $0 <build-dir> <cost-model.csv|-> <out-dir>" >&2; exit 1; }
build="$(cd "$1" && pwd)" cm="$2" out="$3"
here="$(cd "$(dirname "$0")" && pwd)"
enzyme="$(cd "$here/../../.." && pwd)"

so=("$build"/Poseidon/lib/Poseidon-*.so)
[ -e "${so[0]}" ] || { echo "package.sh: no Poseidon-<N>.so in $build/Poseidon/lib" >&2; exit 1; }
v="${so[0]##*/Poseidon-}"; v="${v%.so}"

herbie="$build/Poseidon/herbie/install/herbie"
if [ ! -x "$herbie/bin/herbie" ]; then
  bin="$(sed -n 's/^POSEIDON_HERBIE_BINARY_PATH:INTERNAL=//p' "$build/CMakeCache.txt")"
  herbie="$(cd "$(dirname "$bin")/.." && pwd)"
fi
[ -x "$herbie/bin/herbie" ] || { echo "package.sh: no Herbie install found for $build" >&2; exit 1; }

rm -rf "$out"
mkdir -p "$out/lib" "$out/cost_models" "$out/calibrate"
out="$(cd "$out" && pwd)"
cp "$build/Poseidon/lib/Poseidon-$v.so" "$build/Enzyme/ClangEnzyme-$v.so" \
   "$build/Poseidon/lib/libposeidon_profile.a" \
   "$build/Poseidon/lib/libposeidon_profile_host.a" "$out/lib/"
cp -r "$enzyme/include" "$out/include"
if [ "$cm" != - ]; then
  cp "$cm" "$out/cost_models/cm_x86-64_explorer.csv"
  export POSEIDON_CE_COST_MODEL="$out/cost_models/cm_x86-64_explorer.csv"
elif [ -n "${POSEIDON_CE_COST_MODEL:-}" ]; then
  POSEIDON_CE_COST_MODEL="$(realpath "$POSEIDON_CE_COST_MODEL")"
  export POSEIDON_CE_COST_MODEL
fi
cp "$here/../calibrate/microbm_cpu.py" "$out/calibrate/"
cp -r "$herbie" "$out/herbie"
cp "$here/poseidon-ce.sh" "$here/poseidon-explain-ce.sh" "$here/../explain/poseidon-explain" "$out/"
ln -s poseidon-ce.sh "$out/clang++"
git -C "$enzyme" rev-parse --short=12 HEAD >"$out/VERSION" 2>/dev/null || rm -f "$out/VERSION"

# The examples' Herbie results ship in cache/, so their first compile in the
# explorer does not search. A host search does not read the cost model, so
# they hold under the explorer's own. Herbie writes its scratch files to the
# working directory.
if ! "$out/poseidon-ce.sh" --version >/dev/null 2>&1; then
  echo "package.sh: no clang++-$v to run the examples with; no Herbie results shipped" >&2
elif [ -z "${POSEIDON_CE_COST_MODEL:-}" ]; then
  echo "package.sh: no cost model to run the examples with; no Herbie results shipped" >&2
else
  scratch="$(mktemp -d)"
  cd "$scratch"
  for ex in "$here"/examples/c++/*.cpp; do
    if ! POSEIDON_CE_CACHE="$out/cache" POSEIDON_CE_SOLVE_SECONDS=3600 POSEIDON_CE_EXPLAIN=0 \
      "$out/poseidon-ce.sh" -O2 -S -o /dev/null -poseidon-herbie-timeout=300 \
      -poseidon-herbie-subgraph-timeout=900 "$ex" 2>"$out/warm.log" ||
      grep -q 'program as written\|explorer is busy' "$out/warm.log"; then
      cat "$out/warm.log" >&2
      exit 1
    fi
  done
  rm -rf "$out/warm.log" "$out"/cache/search-slot-* "$out/cache/publish.lock" "$scratch"
fi
