; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

declare i1 @cmp()

define void @f({ double, i32 }* %y, double %x, i32 %z) {
entry:
  %ins = insertvalue { double, i32 } undef, i32 %z, 1
  %ins2 = insertvalue { double, i32 } %ins, double %x, 0
  %e = icmp eq i32 %z, 12
  store { double, i32 } %ins2, { double, i32 }* %y
  ret void
}

declare double @__enzyme_autodiff(...)

define double @test({ double, i32 }* %y, { double, i32 }* %dy, double %x, i32 %z) {
entry:
  %r = call double (...) @__enzyme_autodiff(void ({ double, i32 }*, double, i32)* @f, metadata !"enzyme_dup", { double, i32 }* %y, { double, i32 }* %dy, double %x, i32 %z)
  ret double %r
}

; CHECK: define internal { double } @diffef({ double, i32 }* %y, { double, i32 }* %"y'", double %x, i32 %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { double, i32 }, align 8
; CHECK-NEXT:   %"ins2'de" = alloca { double, i32 }, align 8
; CHECK-NEXT:   store { double, i32 } zeroinitializer, { double, i32 }* %"ins2'de", align 8
; CHECK-NEXT:   %ins = insertvalue { double, i32 } undef, i32 %z, 1
; CHECK-NEXT:   %ins2 = insertvalue { double, i32 } %ins, double %x, 0
; CHECK-NEXT:   store { double, i32 } %ins2, { double, i32 }* %y, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %1 = load { double, i32 }, { double, i32 }* %"y'", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   store { double, i32 } zeroinitializer, { double, i32 }* %0, align 8
; CHECK-NEXT:   %2 = bitcast { double, i32 }* %"y'" to i64*
; CHECK-NEXT:   %3 = bitcast { double, i32 }* %0 to i64*
; CHECK-NEXT:   %4 = load i64, i64* %3, align 4
; CHECK-NEXT:   store i64 %4, i64* %2, align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %5 = extractvalue { double, i32 } %1, 0
; CHECK-NEXT:   %6 = getelementptr inbounds { double, i32 }, { double, i32 }* %"ins2'de", i32 0, i32 0
; CHECK-NEXT:   %7 = load double, double* %6, align 8
; CHECK-NEXT:   %8 = fadd fast double %7, %5
; CHECK-NEXT:   store double %8, double* %6, align 8
; CHECK-NEXT:   %9 = load { double, i32 }, { double, i32 }* %"ins2'de", align 8
; CHECK-NEXT:   %10 = extractvalue { double, i32 } %9, 0
; CHECK-NEXT:   %11 = fadd fast double 0.000000e+00, %10
; CHECK-NEXT:   store { double, i32 } zeroinitializer, { double, i32 }* %"ins2'de", align 8
; CHECK-NEXT:   %12 = insertvalue { double } undef, double %11, 0
; CHECK-NEXT:   ret { double } %12
; CHECK-NEXT: }
