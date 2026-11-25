; The plugin calls only those of Enzyme's helpers that touch no Enzyme state:
; a profile-use compile loads it without Enzyme's plugin, and a symbol only
; Enzyme defines stops the load. No symbol the plugin leaves undefined is one
; that Enzyme's plugin defines and opt does not.
; REQUIRES: enzyme
; RUN: llvm-nm -D --undefined-only --format=just-symbols %poseidon_plugin | env LC_ALL=C sort -u > %t.poseidon
; RUN: llvm-nm -D --defined-only --format=just-symbols %opt | env LC_ALL=C sort -u > %t.host
; RUN: llvm-nm -D --defined-only %enzyme_clang_plugin | awk '$2 ~ /^[TDBR]$/ { print $3 }' | env LC_ALL=C sort -u > %t.enzyme
; RUN: env LC_ALL=C comm -12 %t.poseidon %t.enzyme | env LC_ALL=C comm -23 - %t.host > %t.only-enzyme
; RUN: not grep . %t.only-enzyme
