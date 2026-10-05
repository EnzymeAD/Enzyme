#!/usr/bin/env bash
# Lays out a build the way the nightly-poseidon tarball does:
#   package.sh <build-dir> <cost-model.csv> <out-dir>
set -eu
[ $# -eq 3 ] || { echo "usage: $0 <build-dir> <cost-model.csv> <out-dir>" >&2; exit 1; }
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
mkdir -p "$out/lib" "$out/cost_models"
cp "$build/Poseidon/lib/Poseidon-$v.so" "$build/Enzyme/ClangEnzyme-$v.so" \
   "$build/Poseidon/lib/libposeidon_profile.a" "$out/lib/"
cp -r "$enzyme/include" "$out/include"
cp "$cm" "$out/cost_models/cm_x86-64_explorer.csv"
cp -r "$herbie" "$out/herbie"
cp "$here/poseidon-ce.sh" "$out/"
