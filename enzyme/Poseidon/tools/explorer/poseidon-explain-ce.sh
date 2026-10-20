#!/usr/bin/env bash
# Compiler Explorer tool: presents the explanation a compile left next to its
# output (poseidon-ce.sh, unless POSEIDON_CE_EXPLAIN=0).
#   poseidon-explain-ce.sh <format> [poseidon-explain options...] <source>
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ $# -ge 2 ] || { echo "usage: $0 <format> [options...] <source>" >&2; exit 1; }
format="$1"; shift
src="${!#}"
set -- "${@:1:$(($# - 1))}"
dir="$(dirname "$src")"
if ! ls "$dir"/*.explain.json >/dev/null 2>&1; then
  echo "No Poseidon explanation: the compile failed or was compiled as written (see the compiler output), or it reached no Poseidon site: mark a function with POSEIDON_OPTIMIZE or a call with __poseidon_optimize."
  exit 0
fi
clang=()
for so in "$ROOT"/lib/Poseidon-*.so; do
  v="${so##*/Poseidon-}"; v="${v%.so}"
  [ ! -x "/usr/bin/clang++-$v" ] || clang=(--clang "/usr/bin/clang++-$v")
done
explain=(python3 "$ROOT/poseidon-explain" "$dir" --format "$format" --source "$src" --file-name "${src##*/}"
  "${clang[@]}" "--clang-arg=-I$ROOT/include" "$@")
zw=$'\xe2\x80\x8b'
case "$format" in
  report)
    "${explain[@]}" | sed -E -e "/^[^ :]+:[0-9]+:[0-9]+  /!{/^[[:space:]]*([A-Za-z0-9_.]+|<source>)[(:][0-9]/s/^/$zw/;}" \
      -e 's/^([^ :]+:[0-9]+:[0-9]+)  /\1: note: /' ;;
  diff)
    out="$("${explain[@]}")"
    printf '%s\n' "${out:-Poseidon left the source as written; the report says why.}" ;;
  *) exec "${explain[@]}" ;;
esac
