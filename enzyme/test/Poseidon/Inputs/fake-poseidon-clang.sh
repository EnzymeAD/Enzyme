#!/bin/sh
# Stands in for poseidon-clang++ to poseidon-calibrate's in-kernel arm. Appends
# its arguments to $FAKE_CLANG_LOG when that is set. A compile with
# -poseidon-print prints a candidate table of three in-kernel classes. The
# program it writes records a profile under -poseidon-profile-generate=<dir>,
# else prints a time: 0.5 ms with a rewrite applied, 2 ms without.
[ -z "$FAKE_CLANG_LOG" ] || echo "$*" >> "$FAKE_CLANG_LOG"
out= prev= dir= time=2.000000
for arg; do
  case "$arg" in
  -poseidon-profile-generate=*) dir=${arg#*=} ;;
  -poseidon-apply-rewrites=*) time=0.500000 ;;
  -poseidon-print)
    echo '[poseidon]   #0 wmma m16n16k16 f16/f32  compCost/MAC=1  rel=1  domainError=0' >&2
    echo '[poseidon]   #1 wmma m16n16k16 bf16/f32  compCost/MAC=1  rel=1  domainError=0' >&2
    echo '[poseidon]   #2 tcec n=2 wmma m16n16k8 tf32/f32  compCost/MAC=1  rel=1  domainError=0' >&2 ;;
  esac
  [ "$prev" = -o ] && out=$arg
  prev=$arg
done
[ "$out" = /dev/null ] && exit 0
if [ -n "$dir" ]; then
  printf '#!/bin/sh\nmkdir -p "%s" && : > "%s/matmul_opt.fpprofile"\n' "$dir" "$dir" > "$out"
else
  printf '#!/bin/sh\necho "time_ms = %s"\n' "$time" > "$out"
fi
chmod +x "$out"
