; A profile-use compile loads the plugin without Enzyme's: no symbol it leaves
; undefined is one only Enzyme's plugin defines.
; REQUIRES: enzyme
; RUN: llvm-nm -D --undefined-only --format=just-symbols %poseidon_plugin | env LC_ALL=C sort -u > %t.poseidon
; RUN: llvm-nm -D --defined-only --format=just-symbols %opt | env LC_ALL=C sort -u > %t.host
; RUN: llvm-nm -D --defined-only %enzyme_clang_plugin | awk '$2 ~ /^[TDBR]$/ { print $3 }' | env LC_ALL=C sort -u > %t.enzyme
; RUN: env LC_ALL=C comm -12 %t.poseidon %t.enzyme | env LC_ALL=C comm -23 - %t.host > %t.only-enzyme
; RUN: not grep . %t.only-enzyme
