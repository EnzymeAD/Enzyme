; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg)" -enzyme-preopt=false -S | FileCheck %s

declare [4 x double] @__enzyme_batch(...)


define double @add3(double %x, double %a) {
entry:
  %add = fadd double %x, %a
  ret double %add
}

define double @square(double %x, double %a) {
entry:
  %mul = fmul double %x, %x
  %call = call double @add3(double %mul, double %a)
  ret double %call
}

define [4 x double] @dsquare(double %x1, double %x2, double %x3, double %x4, double %a) {
entry:
  %call = call [4 x double] (...) @__enzyme_batch(double (double, double)* @square, metadata !"enzyme_width", i64 4, metadata !"enzyme_vector", double %x1, double %x2, double %x3, double %x4, metadata !"enzyme_scalar", double %a)
  ret [4 x double] %call
}


; CHECK: define internal [4 x double] @batch_square([4 x double] %x, double %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.x0 = extractvalue [4 x double] %x, 0
; CHECK-NEXT:   %unwrap.x1 = extractvalue [4 x double] %x, 1
; CHECK-NEXT:   %unwrap.x2 = extractvalue [4 x double] %x, 2
; CHECK-NEXT:   %unwrap.x3 = extractvalue [4 x double] %x, 3
; CHECK-NEXT:   %mul0 = fmul double %unwrap.x0, %unwrap.x0
; CHECK-NEXT:   %mul1 = fmul double %unwrap.x1, %unwrap.x1
; CHECK-NEXT:   %mul2 = fmul double %unwrap.x2, %unwrap.x2
; CHECK-NEXT:   %mul3 = fmul double %unwrap.x3, %unwrap.x3
; CHECK-NEXT:   %0 = insertvalue [4 x double] undef, double %mul0, 0
; CHECK-NEXT:   %1 = insertvalue [4 x double] undef, double %mul1, 1
; CHECK-NEXT:   %2 = insertvalue [4 x double] undef, double %mul2, 2
; CHECK-NEXT:   %3 = insertvalue [4 x double] undef, double %mul3, 3
; CHECK-NEXT:   %call = call [4 x double] @batch_add3([4 x double] undef, double %a)
; CHECK-NEXT:   %unwrap.call0 = extractvalue [4 x double] %call, 0
; CHECK-NEXT:   %unwrap.call1 = extractvalue [4 x double] %call, 1
; CHECK-NEXT:   %unwrap.call2 = extractvalue [4 x double] %call, 2
; CHECK-NEXT:   %unwrap.call3 = extractvalue [4 x double] %call, 3
; CHECK-NEXT:   %mrv = insertvalue [4 x double] {{(undef|poison)?}}, double %unwrap.call0, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %unwrap.call1, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %unwrap.call2, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %unwrap.call3, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }

; CHECK: define internal [4 x double] @batch_add3([4 x double] %x, double %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.x0 = extractvalue [4 x double] %x, 0
; CHECK-NEXT:   %unwrap.x1 = extractvalue [4 x double] %x, 1
; CHECK-NEXT:   %unwrap.x2 = extractvalue [4 x double] %x, 2
; CHECK-NEXT:   %unwrap.x3 = extractvalue [4 x double] %x, 3
; CHECK-NEXT:   %add0 = fadd double %unwrap.x0, %a
; CHECK-NEXT:   %add1 = fadd double %unwrap.x1, %a
; CHECK-NEXT:   %add2 = fadd double %unwrap.x2, %a
; CHECK-NEXT:   %add3 = fadd double %unwrap.x3, %a
; CHECK-NEXT:   %mrv = insertvalue [4 x double] {{(undef|poison)?}}, double %add0, 0
; CHECK-NEXT:   %mrv1 = insertvalue [4 x double] %mrv, double %add1, 1
; CHECK-NEXT:   %mrv2 = insertvalue [4 x double] %mrv1, double %add2, 2
; CHECK-NEXT:   %mrv3 = insertvalue [4 x double] %mrv2, double %add3, 3
; CHECK-NEXT:   ret [4 x double] %mrv3
; CHECK-NEXT: }
