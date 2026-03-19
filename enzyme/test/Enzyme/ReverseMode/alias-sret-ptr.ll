; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,gvn,dse)" -S | FileCheck %s

; Subcall takes a pointer output to mimic Array return in memory
define void @subcall(double* %out, double %x) {
entry:
  %m1 = fmul double %x, 2.0
  %m2 = fmul double %x, 3.0
  store double %m1, double* %out, align 8
  %gep = getelementptr inbounds double, double* %out, i64 1
  store double %m2, double* %gep, align 8
  ret void
}

define double @caller(double %x, double* %p) {
entry:
  %call_out = alloca double, i64 2, align 8
  call void @subcall(double* %call_out, double %x)
  
  %val1 = load double, double* %call_out, align 8
  %gep = getelementptr inbounds double, double* %call_out, i64 1
  %val2 = load double, double* %gep, align 8
  
  ; Load carrying alias.scope
  %l = load double, double* %p, align 8, !alias.scope !1
  
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
; CHECK-NEXT:   %"call_out'ipa" = alloca double, i64 2, align 8
; CHECK-NEXT:   %"gep'ipg" = getelementptr inbounds double, double* %"call_out'ipa", i64 1
; CHECK-NEXT:   %[[loadP:.+]] = load double, double* %"p'", align 8, !alias.scope ![[scopeP:[0-9]+]], !noalias ![[noaliasP:[0-9]+]]
; CHECK-NEXT:   %[[addP:.+]] = fadd fast double %[[loadP]], %differeturn
; CHECK-NEXT:   store double %[[addP]], double* %"p'", align 8, !alias.scope ![[scopeP]], !noalias ![[noaliasP]]
; CHECK-NEXT:   store double %differeturn, double* %"gep'ipg", align 8, !alias.scope ![[scopeS:[0-9]+]], !noalias ![[noaliasS:[0-9]+]]
; CHECK-NEXT:   store double %differeturn, double* %"call_out'ipa", align 8, !alias.scope ![[scopeS]], !noalias ![[noaliasS]]
; CHECK-NEXT:   %[[resSubcall:.+]] = call { double } @diffesubcall(double* undef, double* %"call_out'ipa", double %x)
; CHECK-NEXT:   %{{.+}} = extractvalue { double } %[[resSubcall]], 0
; CHECK-NEXT:   ret { double } %[[resSubcall]]
