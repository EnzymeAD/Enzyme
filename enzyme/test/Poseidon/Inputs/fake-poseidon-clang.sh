#!/bin/sh
# Stands in for poseidon-clang++ to poseidon-calibrate's in-kernel arm. Appends
# its arguments to $FAKE_CLANG_LOG when that is set. A compile with
# -poseidon-explain=<dir> explains a product with three in-kernel classes. The
# program it writes records a profile under -poseidon-profile-generate=<dir>,
# else prints a time: 0.5 ms with a rewrite applied, 2 ms without; asked for
# --operand-min <format> <accumulator>, it prints the format's smallest
# subnormal as the floor. With the rewrite $FAKE_CLANG_FAIL (an -poseidon-pin
# value) applied, the program fails as a CUDA program that cannot initialize
# the device does.
[ -z "$FAKE_CLANG_LOG" ] || echo "$*" >> "$FAKE_CLANG_LOG"
out='' prev='' dir='' explain='' time=2.000000 fails=''
for arg; do
  case "$arg" in
  -poseidon-profile-generate=*) dir=${arg#*=} ;;
  -poseidon-pin=*)
    time=0.500000
    [ "${arg#*=}" != "$FAKE_CLANG_FAIL" ] || fails=1 ;;
  -poseidon-explain=*) explain=${arg#*=} ;;
  esac
  [ "$prev" = -o ] && out=$arg
  prev=$arg
done
if [ -n "$explain" ]; then
  mkdir -p "$explain"
  cat > "$explain/matmul_opt.explain.json" <<'EOF'
{"schema": "poseidon-explain/1", "subgraphs": [{"id": "M0", "kind": "matmul",
 "alternatives": [
  {"id": "M0_0", "kind": "matmul", "summary": "wmma m16n16k16 f16/f32", "inkernel_class": "f16_f32"},
  {"id": "M0_1", "kind": "matmul", "summary": "wmma m16n16k16 bf16/f32", "inkernel_class": "bf16_f32"},
  {"id": "M0_2", "kind": "matmul", "summary": "tcec n=2 wmma m16n16k8 tf32/f32", "inkernel_class": "tcec_n2_tf32_f32"},
  {"id": "M0_3", "kind": "matmul", "summary": "ozaki-ii nm=12 wmma m16n16k16 s8/s32"}]}]}
EOF
fi
[ "$out" = /dev/null ] && exit 0
if [ -n "$dir" ]; then
  printf '#!/bin/sh\nmkdir -p "%s" && : > "%s/matmul_opt.fpprofile"\n' "$dir" "$dir" > "$out"
elif [ -n "$fails" ]; then
  printf '#!/bin/sh\necho "CUDA error inkernel.cu:100: initialization error" >&2\nexit 1\n' > "$out"
else
  cat > "$out" <<PROGRAM
#!/bin/sh
case "\$1:\$2" in
--operand-min:f16) echo "operand_min = 5.9604644775390625e-08" ;;
--operand-min:bf16) echo "operand_min = 9.183549615799121e-41" ;;
--operand-min:tf32) echo "operand_min = 1.1479437019748901e-41" ;;
*) echo "time_ms = $time" ;;
esac
PROGRAM
fi
chmod +x "$out"
