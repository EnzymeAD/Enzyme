; DEFINE: %{split} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   "-poseidon-pin=P0=All FP64(50%%) + FP32(50%%)" -S
; RUN: rm -rf %t && mkdir -p %t/exact %t/close %t/apart
; RUN: sed -e 's/MaxRes = inf/MaxRes = 8/' -e 's/, inf, /, 4, /' -e 's/^\tSumSens = .*/\tSumSens = 200/' -e 's/^\tExec = 64$/\tExec = 1000001/' %S/Inputs/unbounded_input/tester.fpprofile > %t/exact/tester.fpprofile
; RUN: sed '6s/.*/\tSumSens = 200.000000002/' %t/exact/tester.fpprofile > %t/close/tester.fpprofile
; RUN: sed '6s/.*/\tSumSens = 200.0002/' %t/exact/tester.fpprofile > %t/apart/tester.fpprofile
; RUN: %{split} -poseidon-profile-use=%t/exact -poseidon-cache=%t/exact/c | FileCheck --check-prefixes=CHECK,TIED %s
; RUN: %{split} -poseidon-profile-use=%t/close -poseidon-cache=%t/close/c | FileCheck --check-prefixes=CHECK,TIED %s
; RUN: %{split} -poseidon-profile-use=%t/apart -poseidon-cache=%t/apart/c | FileCheck --check-prefixes=CHECK,APART %s

; Two operations of 1000001 executions each, whose sensitivities the order of
; the additions moves by up to 1.1e-10 of themselves: 1e-11 apart (close) they
; split as an exact tie does, in the subgraph's order; 1e-6 apart, %b takes the
; narrower tier.

; CHECK-LABEL: define void @tester(double %x, double %y, ptr %o)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %a = fadd double %x, %y
; CHECK-NEXT:    %b = fmul double %a, %y
; CHECK-NEXT:    store double %b, ptr %o, align 8
; CHECK-NEXT:    ret void
; CHECK-LABEL: define void @site(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @preprocess_tester(double %x, double %y, ptr %o)
; CHECK-NEXT:    ret void
define void @tester(double %x, double %y, ptr %o) #0 {
entry:
  %a = fadd double %x, %y
  %b = fmul double %a, %y
  store double %b, ptr %o, align 8
  ret void
}

; TIED-LABEL: define void @preprocess_tester(
; TIED-NEXT:  entry:
; TIED-NEXT:    [[X:%.+]] = fptrunc double %x to float
; TIED-NEXT:    [[Y:%.+]] = fptrunc double %y to float
; TIED-NEXT:    [[A:%.+]] = fadd float [[X]], [[Y]]
; TIED-NEXT:    [[AD:%.+]] = fpext float [[A]] to double
; TIED-NEXT:    [[B:%.+]] = fmul double %y, [[AD]]
; TIED-NEXT:    store double [[B]], ptr %o, align 8
; TIED-NEXT:    ret void

; APART-LABEL: define void @preprocess_tester(
; APART-NEXT:  entry:
; APART-NEXT:    [[A:%.+]] = fadd double %x, %y
; APART-NEXT:    [[AF:%.+]] = fptrunc double [[A]] to float
; APART-NEXT:    [[Y:%.+]] = fptrunc double %y to float
; APART-NEXT:    [[B:%.+]] = fmul float [[Y]], [[AF]]
; APART-NEXT:    [[BD:%.+]] = fpext float [[B]] to double
; APART-NEXT:    store double [[BD]], ptr %o, align 8
; APART-NEXT:    ret void

define void @site(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }
