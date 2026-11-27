; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare { double, double } @csin(double, double)
declare { double, double } @cpow(double, double, double, double)
declare double @carg(double, double)
declare { double, double } @conj(double, double)
declare <2 x float> @csinf(<2 x float>)
declare <2 x float> @cpowf(<2 x float>, <2 x float>)
declare float @cabsf(<2 x float>)

define { double, double } @test_csin(double %x, double %y) {
entry:
  %r = call { double, double } @csin(double %x, double %y)
  ret { double, double } %r
}

define { double, double } @test_cpow(double %ax, double %ay, double %bx, double %by) {
entry:
  %r = call { double, double } @cpow(double %ax, double %ay, double %bx, double %by)
  ret { double, double } %r
}

define double @test_carg(double %x, double %y) {
entry:
  %r = call double @carg(double %x, double %y)
  ret double %r
}

define { double, double } @test_conj(double %x, double %y) {
entry:
  %r = call { double, double } @conj(double %x, double %y)
  ret { double, double } %r
}

define float @test_vec(<2 x float> %a, <2 x float> %b) {
entry:
  %s = call <2 x float> @csinf(<2 x float> %a)
  %p = call <2 x float> @cpowf(<2 x float> %s, <2 x float> %b)
  %m = call float @cabsf(<2 x float> %p)
  ret float %m
}

declare { double, double } @__enzyme_fwddiff2(...)
declare double @__enzyme_fwddiff(...)
declare float @__enzyme_fwddiff_f(...)

define { double, double } @f_csin(double %x, double %dx, double %y, double %dy) {
  %r = call { double, double } (...) @__enzyme_fwddiff2({ double, double } (double, double)* nonnull @test_csin, double %x, double %dx, double %y, double %dy)
  ret { double, double } %r
}

define { double, double } @f_cpow(double %ax, double %dax, double %ay, double %day, double %bx, double %dbx, double %by, double %dby) {
  %r = call { double, double } (...) @__enzyme_fwddiff2({ double, double } (double, double, double, double)* nonnull @test_cpow, double %ax, double %dax, double %ay, double %day, double %bx, double %dbx, double %by, double %dby)
  ret { double, double } %r
}

define double @f_carg(double %x, double %dx, double %y, double %dy) {
  %r = call double (...) @__enzyme_fwddiff(double (double, double)* nonnull @test_carg, double %x, double %dx, double %y, double %dy)
  ret double %r
}

define { double, double } @f_conj(double %x, double %dx, double %y, double %dy) {
  %r = call { double, double } (...) @__enzyme_fwddiff2({ double, double } (double, double)* nonnull @test_conj, double %x, double %dx, double %y, double %dy)
  ret { double, double } %r
}

define float @f_vec(<2 x float> %a, <2 x float> %da, <2 x float> %b, <2 x float> %db) {
  %r = call float (...) @__enzyme_fwddiff_f(float (<2 x float>, <2 x float>)* nonnull @test_vec, <2 x float> %a, <2 x float> %da, <2 x float> %b, <2 x float> %db)
  ret float %r
}

; CHECK: define internal { double, double } @fwddiffetest_csin(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[COS:.+]] = call { double, double } @ccos(double %x, double %y)
; CHECK-NEXT:   %[[CRE:.+]] = extractvalue { double, double } %[[COS]], 0
; CHECK-NEXT:   %[[CIM:.+]] = extractvalue { double, double } %[[COS]], 1
; CHECK-NEXT:   %[[T0:.+]] = fmul fast double %[[CRE]], %"x'"
; CHECK-NEXT:   %[[T1:.+]] = fmul fast double %[[CIM]], %"y'"
; CHECK-NEXT:   %[[RE:.+]] = fsub fast double %[[T0]], %[[T1]]
; CHECK-NEXT:   %[[T2:.+]] = fmul fast double %[[CRE]], %"y'"
; CHECK-NEXT:   %[[T3:.+]] = fmul fast double %[[CIM]], %"x'"
; CHECK-NEXT:   %[[IM:.+]] = fadd fast double %[[T2]], %[[T3]]
; CHECK-NEXT:   %[[R0:.+]] = insertvalue { double, double } undef, double %[[RE]], 0
; CHECK-NEXT:   %[[R1:.+]] = insertvalue { double, double } %[[R0]], double %[[IM]], 1
; CHECK-NEXT:   ret { double, double } %[[R1]]

; CHECK: define internal { double, double } @fwddiffetest_cpow(double %ax, double %"ax'", double %ay, double %"ay'", double %bx, double %"bx'", double %by, double %"by'")
; CHECK: call { double, double } @cpow(double %ax, double %ay,
; CHECK: call { double, double } @clog(double %ax, double %ay)
; CHECK: ret { double, double }

; CHECK: define internal double @fwddiffetest_carg(double %x, double %"x'", double %y, double %"y'")
; CHECK: fdiv fast double
; CHECK: ret double

; CHECK: define internal { double, double } @fwddiffetest_conj(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[NEG:.+]] = fneg fast double %"y'"
; CHECK-NEXT:   %[[C0:.+]] = insertvalue { double, double } undef, double %"x'", 0
; CHECK-NEXT:   %[[C1:.+]] = insertvalue { double, double } %[[C0]], double %[[NEG]], 1
; CHECK-NEXT:   ret { double, double } %[[C1]]

; CHECK: define internal float @fwddiffetest_vec(<2 x float> %a, <2 x float> %"a'", <2 x float> %b, <2 x float> %"b'")
; CHECK: call fast <2 x float> @ccosf(<2 x float>
; CHECK: call fast <2 x float> @clogf(<2 x float>
; CHECK: call fast float @cabsf(<2 x float>
; CHECK: ret float
