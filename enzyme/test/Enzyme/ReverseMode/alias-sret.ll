; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg)" -S | FileCheck %s; fi

define { double, double } @subcall(double %x) {
entry:
  %m1 = fmul double %x, 2.0
  %m2 = fmul double %x, 3.0
  %res1 = insertvalue { double, double } undef, double %m1, 0
  %res2 = insertvalue { double, double } %res1, double %m2, 1
  ret { double, double } %res2
}

define double @caller(double %x, double* %p) {
entry:
  %call = call { double, double } @subcall(double %x)
  %val1 = extractvalue { double, double } %call, 0
  %val2 = extractvalue { double, double } %call, 1
  %l = load double, double* %p, !alias.scope !1
  %res = fadd double %val1, %val2
  %res2 = fadd double %res, %l
  ret double %res2
}

!1 = !{!2}
!2 = distinct !{!2, !3, !"domain"}
!3 = distinct !{!3, !"domain"}

define void @test_diff(double %x, double* %p, double* %dp) {
  call void (...) @__enzyme_autodiff(i8* bitcast (double (double, double*)* @caller to i8*), double %x, double* %p, double* %dp)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffecaller(double %x, double* nocapture readonly %p, double* nocapture %"p'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"call'de" = alloca { double, double }, align 8
; CHECK-NEXT:   store { double, double } zeroinitializer, { double, double }* %"call'de", align 8
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:
; CHECK-NEXT:   %[[add0:.+]] = fadd fast double 0.000000e+00, %differeturn
; CHECK-NEXT:   %[[add1:.+]] = fadd fast double 0.000000e+00, %differeturn
; CHECK-NEXT:   %[[add2:.+]] = fadd fast double 0.000000e+00, %[[add0]]
; CHECK-NEXT:   %[[add3:.+]] = fadd fast double 0.000000e+00, %[[add0]]
; CHECK-NEXT:   %[[loadP:.+]] = load double, double* %"p'", align 8, !alias.scope ![[scopeP:[0-9]+]], !noalias ![[noaliasP:[0-9]+]]
; CHECK-NEXT:   %[[addP:.+]] = fadd fast double %[[loadP]], %[[add1]]
; CHECK-NEXT:   store double %[[addP]], double* %"p'", align 8, !alias.scope ![[scopeP]], !noalias ![[noaliasP]]
; CHECK-NEXT:   %[[gepS:.+]] = getelementptr inbounds { double, double }, { double, double }* %"call'de", i32 0, i32 1
; CHECK-NEXT:   %[[loadS:.+]] = load double, double* %[[gepS]], align 8
; CHECK-NEXT:   %[[addS:.+]] = fadd fast double %[[loadS]], %[[add3]]
; CHECK-NEXT:   store double %[[addS]], double* %[[gepS]], align 8{{$}}
