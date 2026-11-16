; The accuracy model runs on one thread unless -poseidon-threads asks for
; more; 0 asks for every hardware thread.
; RUN: %opt %s %loadPoseidon -passes=verify --print-all-options -disable-output | FileCheck %s
; RUN: %opt %s %loadPoseidon -passes=verify -poseidon-threads=0 --print-all-options -disable-output | FileCheck --check-prefix=ALL %s

; CHECK: --poseidon-threads {{ *}}= 1 {{ *}}(default: 1)
; ALL: --poseidon-threads {{ *}}= 0 {{ *}}(default: 1)

define void @f() {
  ret void
}
