; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

declare i1 @cmp()

define void @f({ double, i1 }* %y, double %x, i1 %z) {
entry:
  %ins = insertvalue { double, i1 } undef, double %x, 0
  %ins2 = insertvalue { double, i1 } %ins, i1 %z, 1
  store { double, i1 } %ins2, { double, i1 }* %y
  ret void
}

declare double @__enzyme_autodiff(...)

define double @test({ double, i32 }* %y, { double, i32 }* %dy, double %x, i1 %z) {
entry:
  %r = call double (...) @__enzyme_autodiff(void ({ double, i1 }*, double, i1)* @f, metadata !"enzyme_dup", { double, i32 }* %y, { double, i32 }* %dy, double %x, i1 %z)
  ret double %r
}

; CHECK: define internal { double } @diffef({ double, i1 }* %y, { double, i1 }* %"y'", double %x, i1 %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"ins2'de" = alloca { double, i1 }, align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins2'de", align 8
; CHECK-NEXT:   %"ins'de" = alloca { double, i1 }, align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins'de", align 8
; CHECK-NEXT:   %ins = insertvalue { double, i1 } undef, double %x, 0
; CHECK-NEXT:   %ins2 = insertvalue { double, i1 } %ins, i1 %z, 1
; CHECK-NEXT:   store { double, i1 } %ins2, { double, i1 }* %y, align 8
; CHECK-NEXT:   %0 = load { double, i1 }, { double, i1 }* %"y'", align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"y'", align 8
; CHECK-NEXT:   %1 = load { double, i1 }, { double, i1 }* %"ins2'de", align 8
; CHECK-NEXT:   %2 = extractvalue { double, i1 } %0, 0
; CHECK-NEXT:   %3 = getelementptr inbounds { double, i1 }, { double, i1 }* %"ins2'de", i32 0, i32 0
; CHECK-NEXT:   %4 = load double, double* %3, align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %2
; CHECK-NEXT:   store double %5, double* %3, align 8
; CHECK-NEXT:   %6 = load { double, i1 }, { double, i1 }* %"ins2'de", align 8
; CHECK-NEXT:   %7 = load { double, i1 }, { double, i1 }* %"ins'de", align 8
; CHECK-NEXT:   %8 = extractvalue { double, i1 } %6, 0
; CHECK-NEXT:   %9 = getelementptr inbounds { double, i1 }, { double, i1 }* %"ins'de", i32 0, i32 0
; CHECK-NEXT:   %10 = load double, double* %9, align 8
; CHECK-NEXT:   %11 = fadd fast double %10, %8
; CHECK-NEXT:   store double %11, double* %9, align 8
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins2'de", align 8
; CHECK-NEXT:   %12 = load { double, i1 }, { double, i1 }* %"ins'de", align 8
; CHECK-NEXT:   %13 = extractvalue { double, i1 } %12, 0
; CHECK-NEXT:   %14 = fadd fast double 0.000000e+00, %13
; CHECK-NEXT:   store { double, i1 } zeroinitializer, { double, i1 }* %"ins'de", align 8
; CHECK-NEXT:   %15 = insertvalue { double } undef, double %14, 0
; CHECK-NEXT:   ret { double } %15
; CHECK-NEXT: }
