; -help lists the flags a user passes, under their own heading; every other
; flag is listed by -help-hidden only.
; RUN: %opt %loadPoseidon -help | FileCheck %s
; RUN: %opt %loadPoseidon -help-hidden | FileCheck --check-prefix=HIDDEN %s

; CHECK-LABEL: Poseidon options:
; CHECK-NOT: -poseidon-{{[a-z-]+}}
; CHECK: --poseidon-cache=<directory>
; CHECK-NEXT: --poseidon-cost-budget=<share>
; CHECK-NEXT: --poseidon-cost-model=<csv>
; CHECK-NEXT: --poseidon-explain=<directory>
; CHECK: --poseidon-kernels=<regex>
; CHECK-NEXT: --poseidon-print
; CHECK-NEXT: --poseidon-profile-generate
; CHECK-NEXT: --poseidon-profile-use[=<directory>]
; CHECK-NEXT: --poseidon-report-path=<directory>
; CHECK: --poseidon-tau=<error>
; CHECK-NOT: --poseidon-

; HIDDEN-LABEL: Poseidon options:
; HIDDEN: --poseidon-herbie-pre-floor-bits=
; HIDDEN: --poseidon-herbie-processes=
; HIDDEN: --poseidon-tau=<error>
