; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-activity-analysis -activity-analysis-func=wrapper -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=wrapper -S | FileCheck %s

; A wrapper passes its own sret straight through to a local read-only-or-throw
; callee, as call-slot optimization leaves it. The sret memory belongs to the
; caller and is active, so the call is active: it used to be found constant
; because the sret argument's only user within the wrapper is that call.

define void @inner(double* noalias nocapture writeonly sret(double) %out, double* nocapture readonly %p) #0 {
entry:
  %v = load double, double* %p
  %m = fmul double %v, 2.000000e+00
  store double %m, double* %out
  ret void
}

define void @wrapper(double* noalias nocapture writeonly sret(double) %out, double* nocapture readonly %p) {
entry:
  call void @inner(double* sret(double) %out, double* %p)
  ret void
}

attributes #0 = { "enzyme_LocalReadOnlyOrThrow" }

; CHECK: {{double\*|ptr}} %out: icv:0
; CHECK-NEXT: {{double\*|ptr}} %p: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   call void @inner({{double\*|ptr}} sret(double) %out, {{double\*|ptr}} %p): icv:1 ici:0
; CHECK-NEXT:   ret void: icv:1 ici:1
