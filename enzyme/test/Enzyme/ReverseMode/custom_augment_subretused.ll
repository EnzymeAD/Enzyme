; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -O2 -S | %lli - ; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" -enzyme-preopt=false -S | %lli -

@enzyme_dup = external global i32
@enzyme_out = external global i32
@enzyme_primal_return = external global i32
@sink = global double 0.000000e+00

declare void @__enzyme_autodiff_void(ptr, ...)
declare { double, double } @__enzyme_autodiff_ret(ptr, ...)

define double @aug_f_readnone(double %x) {
entry:
  %sq = fmul double %x, %x
  ret double %sq
}

define double @rev_f_readnone(double %x, double %dret) {
entry:
  %two_x = fmul double 2.000000e+00, %x
  %grad = fmul double %two_x, %dret
  ret double %grad
}

define double @f_readnone(double %x) #0 !enzyme_augment !0 !enzyme_gradient !1 {
entry:
  %sq = fmul double %x, %x
  ret double %sq
}

define double @aug_f_mem(double %x) {
entry:
  %sq = fmul double %x, %x
  %cb = fmul double %sq, %x
  ret double %cb
}

define double @rev_f_mem(double %x, double %dret) {
entry:
  %sq = fmul double %x, %x
  %three_sq = fmul double 3.000000e+00, %sq
  %grad = fmul double %three_sq, %dret
  ret double %grad
}

define double @f_mem(double %x) !enzyme_augment !2 !enzyme_gradient !3 {
entry:
  store double %x, ptr @sink, align 8
  %sq = fmul double %x, %x
  %cb = fmul double %sq, %x
  ret double %cb
}

define void @caller_readnone(ptr %x) {
entry:
  %x0 = load double, ptr %x, align 8
  %p1 = getelementptr inbounds double, ptr %x, i64 1
  %x1 = load double, ptr %p1, align 8
  %s = fadd double %x0, %x1
  %r = call double @f_readnone(double %s)
  %p2 = getelementptr inbounds double, ptr %x, i64 2
  store double %r, ptr %p2, align 8
  ret void
}

define void @caller_mem(ptr %x) {
entry:
  %x0 = load double, ptr %x, align 8
  %p1 = getelementptr inbounds double, ptr %x, i64 1
  %x1 = load double, ptr %p1, align 8
  %s = fadd double %x0, %x1
  %r = call double @f_mem(double %s)
  %p2 = getelementptr inbounds double, ptr %x, i64 2
  store double %r, ptr %p2, align 8
  ret void
}

define i32 @main() {
entry:
  %x = alloca [3 x double], align 8
  %dx = alloca [3 x double], align 8
  %x0p = getelementptr inbounds [3 x double], ptr %x, i64 0, i64 0
  %x1p = getelementptr inbounds [3 x double], ptr %x, i64 0, i64 1
  %x2p = getelementptr inbounds [3 x double], ptr %x, i64 0, i64 2
  %dx0p = getelementptr inbounds [3 x double], ptr %dx, i64 0, i64 0
  %dx1p = getelementptr inbounds [3 x double], ptr %dx, i64 0, i64 1
  %dx2p = getelementptr inbounds [3 x double], ptr %dx, i64 0, i64 2

  store double 2.000000e+00, ptr %x0p, align 8
  store double 3.000000e+00, ptr %x1p, align 8
  store double 0.000000e+00, ptr %x2p, align 8
  store double 0.000000e+00, ptr %dx0p, align 8
  store double 0.000000e+00, ptr %dx1p, align 8
  store double 1.000000e+00, ptr %dx2p, align 8

  %dup = load i32, ptr @enzyme_dup, align 4
  call void (ptr, ...) @__enzyme_autodiff_void(ptr @caller_readnone, i32 %dup, ptr %x0p, ptr %dx0p)

  %r_rn = load double, ptr %x2p, align 8
  %dx0_rn = load double, ptr %dx0p, align 8
  %dx1_rn = load double, ptr %dx1p, align 8
  %ok1 = fcmp oeq double %r_rn, 2.500000e+01
  %ok2 = fcmp oeq double %dx0_rn, 1.000000e+01
  %ok3 = fcmp oeq double %dx1_rn, 1.000000e+01
  %ok12 = and i1 %ok1, %ok2
  %ok123 = and i1 %ok12, %ok3
  br i1 %ok123, label %test_mem, label %fail

test_mem:
  store double 2.000000e+00, ptr %x0p, align 8
  store double 3.000000e+00, ptr %x1p, align 8
  store double 0.000000e+00, ptr %x2p, align 8
  store double 0.000000e+00, ptr %dx0p, align 8
  store double 0.000000e+00, ptr %dx1p, align 8
  store double 1.000000e+00, ptr %dx2p, align 8

  call void (ptr, ...) @__enzyme_autodiff_void(ptr @caller_mem, i32 %dup, ptr %x0p, ptr %dx0p)

  %r_mem = load double, ptr %x2p, align 8
  %dx0_mem = load double, ptr %dx0p, align 8
  %dx1_mem = load double, ptr %dx1p, align 8
  %ok4 = fcmp oeq double %r_mem, 1.250000e+02
  %ok5 = fcmp oeq double %dx0_mem, 7.500000e+01
  %ok6 = fcmp oeq double %dx1_mem, 7.500000e+01
  %ok45 = and i1 %ok4, %ok5
  %ok456 = and i1 %ok45, %ok6
  br i1 %ok456, label %test_top, label %fail

test_top:
  %pret = load i32, ptr @enzyme_primal_return, align 4
  %out = load i32, ptr @enzyme_out, align 4
  %top_res = call { double, double } (ptr, ...) @__enzyme_autodiff_ret(ptr @f_readnone, i32 %pret, i32 %out, double 4.000000e+00)
  %top_primal = extractvalue { double, double } %top_res, 0
  %top_grad = extractvalue { double, double } %top_res, 1
  %ok7 = fcmp oeq double %top_primal, 1.600000e+01
  %ok8 = fcmp oeq double %top_grad, 8.000000e+00
  %ok78 = and i1 %ok7, %ok8
  br i1 %ok78, label %pass, label %fail

pass:
  ret i32 0

fail:
  ret i32 1
}

attributes #0 = { nounwind readnone willreturn }

!0 = !{ptr @aug_f_readnone}
!1 = !{ptr @rev_f_readnone}
!2 = !{ptr @aug_f_mem}
!3 = !{ptr @rev_f_mem}

; CHECK: define internal void @diffecaller_mem(ptr nocapture %x, ptr nocapture %"x'")
; CHECK:   %r = call fast double @aug_f_mem(double %s)
; CHECK:   store double %r, ptr %p2

; CHECK: define internal { double, double } @fixgradient_f_readnone.2(double %arg0, double %arg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[AUG:.+]] = call double @aug_f_readnone(double %arg0)
; CHECK-NEXT:   %[[REV:.+]] = call { double } @fixgradient_f_readnone.{{[0-9]+}}(double %arg0, double %arg1)
; CHECK-NEXT:   %[[INS0:.+]] = insertvalue { double, double } undef, double %[[AUG]], 0
; CHECK-NEXT:   %[[EXT:.+]] = extractvalue { double } %[[REV]], 0
; CHECK-NEXT:   %[[INS1:.+]] = insertvalue { double, double } %[[INS0]], double %[[EXT]], 1
; CHECK-NEXT:   ret { double, double } %[[INS1]]
; CHECK-NEXT: }
