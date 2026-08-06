; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -opaque-pointers=1 -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -opaque-pointers=1 -S | FileCheck %s

%Node = type { double, ptr }

@g = internal global %Node { double 1.0, ptr getelementptr inbounds (%Node, ptr @g, i32 0, i32 0) }, align 8

define double @tester(ptr %x, double %y) {
entry:
  %ptr = load ptr, ptr getelementptr inbounds (%Node, ptr @g, i32 0, i32 1), align 8
  %v = load double, ptr %x, align 8
  %mul = fmul double %v, %y
  ret double %mul
}

define void @dtester(ptr %x, ptr %dx, double %y) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @tester, ptr %x, ptr %dx, double %y)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

; CHECK: define internal { double } @diffetester(ptr %x, ptr %"x'", double %y, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %v = load double, ptr %x, align 8
; CHECK-NEXT:   %0 = fmul fast double %differeturn, %y
; CHECK-NEXT:   %1 = fmul fast double %differeturn, %v
; CHECK-NEXT:   %2 = load double, ptr %"x'", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %0
; CHECK-NEXT:   store double %3, ptr %"x'", align 8
; CHECK-NEXT:   %4 = insertvalue { double } undef, double %1, 0
; CHECK-NEXT:   ret { double } %4
; CHECK-NEXT: }
