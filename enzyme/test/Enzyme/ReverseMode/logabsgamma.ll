; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x) {
entry:
  %a = call { double, i64 } @logabsgamma(double %x)
  %b = extractvalue { double, i64 } %a, 0 
  ret double %b
}

define double @test_derivative(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @tester, double %x)
  ret double %0
}

declare { double, i64 } @logabsgamma(double)

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double)*, ...)

; CHECK: define internal { double } @diffetester(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"a'de" = alloca { double, i64 }, align 8
; CHECK-NEXT:   store { double, i64 } zeroinitializer, { double, i64 }* %"a'de", align 8
; CHECK-NEXT:   %0 = getelementptr inbounds { double, i64 }, { double, i64 }* %"a'de", i32 0, i32 0
; CHECK-NEXT:   %1 = load double, double* %0, align 8
; CHECK-NEXT:   %2 = fadd fast double %1, %differeturn
; CHECK-NEXT:   store double %2, double* %0, align 8
; CHECK-NEXT:   %3 = load { double, i64 }, { double, i64 }* %"a'de", align 8
; CHECK-NEXT:   store { double, i64 } zeroinitializer, { double, i64 }* %"a'de", align 8
; CHECK-NEXT:   %4 = call fast double @digamma(double %x)
; CHECK-NEXT:   %5 = extractvalue { double, i64 } %3, 0
; CHECK-NEXT:   %6 = fmul fast double %4, %5
; CHECK-NEXT:   %7 = insertvalue { double } undef, double %6, 0
; CHECK-NEXT:   ret { double } %7
; CHECK-NEXT: }
