; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,globaldce" -S | %lli -; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="default<O2>,enzyme,default<O2>" -S | %lli -; fi

declare double @llvm.maximum.f64(double, double)
declare double @llvm.minimum.f64(double, double)
declare <2 x double> @llvm.maximum.v2f64(<2 x double>, <2 x double>)
declare <2 x double> @llvm.minimum.v2f64(<2 x double>, <2 x double>)

declare { double } @__enzyme_autodiff_1(...)
declare { double, double } @__enzyme_autodiff_2(...)
declare { <2 x double>, <2 x double> } @__enzyme_autodiff_v2(...)
declare double @__enzyme_fwddiff(...)
declare { double, double } @__enzyme_fwddiff_w2(...)
declare <2 x double> @__enzyme_fwddiff_v2(...)

define internal double @relu_diff(double %x) {
entry:
  %a = call double @llvm.maximum.f64(double %x, double 0.0)
  %nx = fneg double %x
  %b = call double @llvm.maximum.f64(double %nx, double 0.0)
  %r = fsub double %a, %b
  ret double %r
}

define internal double @min_diff(double %x) {
entry:
  %a = call double @llvm.minimum.f64(double %x, double 0.0)
  %nx = fneg double %x
  %b = call double @llvm.minimum.f64(double %nx, double 0.0)
  %r = fsub double %a, %b
  ret double %r
}

define internal double @fmax_fn(double %x, double %y) {
entry:
  %r = call double @llvm.maximum.f64(double %x, double %y)
  ret double %r
}

define internal double @fmin_fn(double %x, double %y) {
entry:
  %r = call double @llvm.minimum.f64(double %x, double %y)
  ret double %r
}

define internal <2 x double> @fmax_v2(<2 x double> %x, <2 x double> %y) {
entry:
  %r = call <2 x double> @llvm.maximum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %r
}

define internal <2 x double> @fmin_v2(<2 x double> %x, <2 x double> %y) {
entry:
  %r = call <2 x double> @llvm.minimum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %r
}

define internal i32 @check_scalar(double %got, double %exp, i32 %code) {
entry:
  %ok = fcmp oeq double %got, %exp
  %res = select i1 %ok, i32 0, i32 %code
  ret i32 %res
}

define internal i32 @check_pair_max(double %x, double %y, double %ex, double %ey, i32 %base) noinline {
entry:
  %rev = call { double, double } (...) @__enzyme_autodiff_2(ptr @fmax_fn, double %x, double %y)
  %rx = extractvalue { double, double } %rev, 0
  %ry = extractvalue { double, double } %rev, 1
  %ok_rx = fcmp oeq double %rx, %ex
  %ok_ry = fcmp oeq double %ry, %ey
  %ok_rev = and i1 %ok_rx, %ok_ry
  br i1 %ok_rev, label %chk_fwd, label %fail_rev

fail_rev:
  ret i32 %base

chk_fwd:
  %fx = call double (...) @__enzyme_fwddiff(ptr @fmax_fn, double %x, double 1.0, double %y, double 0.0)
  %fy = call double (...) @__enzyme_fwddiff(ptr @fmax_fn, double %x, double 0.0, double %y, double 1.0)
  %ok_fx = fcmp oeq double %fx, %ex
  %ok_fy = fcmp oeq double %fy, %ey
  %ok_fwd = and i1 %ok_fx, %ok_fy
  br i1 %ok_fwd, label %chk_w2, label %fail_fwd

fail_fwd:
  %c1 = add i32 %base, 1
  ret i32 %c1

chk_w2:
  %w2 = call { double, double } (...) @__enzyme_fwddiff_w2(ptr @fmax_fn, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  %wx = extractvalue { double, double } %w2, 0
  %wy = extractvalue { double, double } %w2, 1
  %ok_wx = fcmp oeq double %wx, %ex
  %ok_wy = fcmp oeq double %wy, %ey
  %ok_w2 = and i1 %ok_wx, %ok_wy
  %c2 = add i32 %base, 2
  %retv = select i1 %ok_w2, i32 0, i32 %c2
  ret i32 %retv
}

define internal i32 @check_pair_min(double %x, double %y, double %ex, double %ey, i32 %base) noinline {
entry:
  %rev = call { double, double } (...) @__enzyme_autodiff_2(ptr @fmin_fn, double %x, double %y)
  %rx = extractvalue { double, double } %rev, 0
  %ry = extractvalue { double, double } %rev, 1
  %ok_rx = fcmp oeq double %rx, %ex
  %ok_ry = fcmp oeq double %ry, %ey
  %ok_rev = and i1 %ok_rx, %ok_ry
  br i1 %ok_rev, label %chk_fwd, label %fail_rev

fail_rev:
  ret i32 %base

chk_fwd:
  %fx = call double (...) @__enzyme_fwddiff(ptr @fmin_fn, double %x, double 1.0, double %y, double 0.0)
  %fy = call double (...) @__enzyme_fwddiff(ptr @fmin_fn, double %x, double 0.0, double %y, double 1.0)
  %ok_fx = fcmp oeq double %fx, %ex
  %ok_fy = fcmp oeq double %fy, %ey
  %ok_fwd = and i1 %ok_fx, %ok_fy
  br i1 %ok_fwd, label %chk_w2, label %fail_fwd

fail_fwd:
  %c1 = add i32 %base, 1
  ret i32 %c1

chk_w2:
  %w2 = call { double, double } (...) @__enzyme_fwddiff_w2(ptr @fmin_fn, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 0.0, double %y, double 0.0, double 1.0)
  %wx = extractvalue { double, double } %w2, 0
  %wy = extractvalue { double, double } %w2, 1
  %ok_wx = fcmp oeq double %wx, %ex
  %ok_wy = fcmp oeq double %wy, %ey
  %ok_w2 = and i1 %ok_wx, %ok_wy
  %c2 = add i32 %base, 2
  %retv = select i1 %ok_w2, i32 0, i32 %c2
  ret i32 %retv
}

define i32 @main() {
entry:
  ; Issue #3279: relu(x) - relu(-x) and min(x, 0) - min(-x, 0) at +0.0, -0.0, +0.5, -0.5.
  %r1s = call { double } (...) @__enzyme_autodiff_1(ptr @relu_diff, double 0.0)
  %r1 = extractvalue { double } %r1s, 0
  %e1 = call i32 @check_scalar(double %r1, double 1.0, i32 1)
  %b1 = icmp ne i32 %e1, 0
  br i1 %b1, label %fail, label %t2

t2:
  %r2s = call { double } (...) @__enzyme_autodiff_1(ptr @relu_diff, double -0.0)
  %r2 = extractvalue { double } %r2s, 0
  %e2 = call i32 @check_scalar(double %r2, double 1.0, i32 2)
  %b2 = icmp ne i32 %e2, 0
  br i1 %b2, label %fail, label %t3

t3:
  %f1 = call double (...) @__enzyme_fwddiff(ptr @relu_diff, double 0.0, double 1.0)
  %e3 = call i32 @check_scalar(double %f1, double 1.0, i32 3)
  %b3 = icmp ne i32 %e3, 0
  br i1 %b3, label %fail, label %t4

t4:
  %f2 = call double (...) @__enzyme_fwddiff(ptr @relu_diff, double -0.0, double 1.0)
  %e4 = call i32 @check_scalar(double %f2, double 1.0, i32 4)
  %b4 = icmp ne i32 %e4, 0
  br i1 %b4, label %fail, label %t5

t5:
  %r5s = call { double } (...) @__enzyme_autodiff_1(ptr @min_diff, double 0.0)
  %r5 = extractvalue { double } %r5s, 0
  %e5 = call i32 @check_scalar(double %r5, double 1.0, i32 5)
  %b5 = icmp ne i32 %e5, 0
  br i1 %b5, label %fail, label %t6

t6:
  %r6s = call { double } (...) @__enzyme_autodiff_1(ptr @min_diff, double -0.0)
  %r6 = extractvalue { double } %r6s, 0
  %e6 = call i32 @check_scalar(double %r6, double 1.0, i32 6)
  %b6 = icmp ne i32 %e6, 0
  br i1 %b6, label %fail, label %t7

t7:
  %f7 = call double (...) @__enzyme_fwddiff(ptr @min_diff, double 0.0, double 1.0)
  %e7 = call i32 @check_scalar(double %f7, double 1.0, i32 7)
  %b7 = icmp ne i32 %e7, 0
  br i1 %b7, label %fail, label %t8

t8:
  %f8 = call double (...) @__enzyme_fwddiff(ptr @min_diff, double -0.0, double 1.0)
  %e8 = call i32 @check_scalar(double %f8, double 1.0, i32 8)
  %b8 = icmp ne i32 %e8, 0
  br i1 %b8, label %fail, label %t_pairs

t_pairs:
  %pnan = bitcast i64 9221120237041090560 to double
  %nnan = bitcast i64 -2251799813685248 to double
  %pinf = bitcast i64 9218868437227405312 to double
  %ninf = bitcast i64 -4503599627370496 to double

  ; maximum cases
  %c10 = call i32 @check_pair_max(double -0.0, double 0.0, double 0.0, double 1.0, i32 10)
  %c11 = call i32 @check_pair_max(double 0.0, double -0.0, double 1.0, double 0.0, i32 13)
  %c12 = call i32 @check_pair_max(double 0.0, double 0.0, double 1.0, double 0.0, i32 16)
  %c13 = call i32 @check_pair_max(double -0.0, double -0.0, double 1.0, double 0.0, i32 19)
  %c14 = call i32 @check_pair_max(double 1.0, double 2.0, double 0.0, double 1.0, i32 22)
  %c15 = call i32 @check_pair_max(double 2.0, double 1.0, double 1.0, double 0.0, i32 25)
  %c16 = call i32 @check_pair_max(double 2.0, double 2.0, double 1.0, double 0.0, i32 28)
  %c17 = call i32 @check_pair_max(double 2.0, double %pnan, double 0.0, double 1.0, i32 31)
  %c18 = call i32 @check_pair_max(double 2.0, double %nnan, double 0.0, double 1.0, i32 34)
  %c19 = call i32 @check_pair_max(double %pnan, double 2.0, double 1.0, double 0.0, i32 37)
  %c20 = call i32 @check_pair_max(double %nnan, double 2.0, double 1.0, double 0.0, i32 40)
  %c21 = call i32 @check_pair_max(double %pnan, double %nnan, double 1.0, double 0.0, i32 43)
  %c22 = call i32 @check_pair_max(double %ninf, double %pinf, double 0.0, double 1.0, i32 46)
  %c23 = call i32 @check_pair_max(double %pinf, double %ninf, double 1.0, double 0.0, i32 49)

  ; minimum cases
  %c30 = call i32 @check_pair_min(double -0.0, double 0.0, double 1.0, double 0.0, i32 60)
  %c31 = call i32 @check_pair_min(double 0.0, double -0.0, double 0.0, double 1.0, i32 63)
  %c32 = call i32 @check_pair_min(double 0.0, double 0.0, double 0.0, double 1.0, i32 66)
  %c33 = call i32 @check_pair_min(double -0.0, double -0.0, double 0.0, double 1.0, i32 69)
  %c34 = call i32 @check_pair_min(double 1.0, double 2.0, double 1.0, double 0.0, i32 72)
  %c35 = call i32 @check_pair_min(double 2.0, double 1.0, double 0.0, double 1.0, i32 75)
  %c36 = call i32 @check_pair_min(double 2.0, double 2.0, double 0.0, double 1.0, i32 78)
  %c37 = call i32 @check_pair_min(double %pnan, double 2.0, double 1.0, double 0.0, i32 81)
  %c38 = call i32 @check_pair_min(double %nnan, double 2.0, double 1.0, double 0.0, i32 84)
  %c39 = call i32 @check_pair_min(double 2.0, double %pnan, double 0.0, double 1.0, i32 87)
  %c40 = call i32 @check_pair_min(double 2.0, double %nnan, double 0.0, double 1.0, i32 90)
  %c41 = call i32 @check_pair_min(double %pnan, double %nnan, double 0.0, double 1.0, i32 93)
  %c42 = call i32 @check_pair_min(double %ninf, double %pinf, double 1.0, double 0.0, i32 96)
  %c43 = call i32 @check_pair_min(double %pinf, double %ninf, double 0.0, double 1.0, i32 99)

  %s0 = or i32 %c10, %c11
  %s1 = or i32 %s0, %c12
  %s2 = or i32 %s1, %c13
  %s3 = or i32 %s2, %c14
  %s4 = or i32 %s3, %c15
  %s5 = or i32 %s4, %c16
  %s6 = or i32 %s5, %c17
  %s7 = or i32 %s6, %c18
  %s8 = or i32 %s7, %c19
  %s9 = or i32 %s8, %c20
  %s10 = or i32 %s9, %c21
  %s11 = or i32 %s10, %c22
  %s12 = or i32 %s11, %c23
  %s13 = or i32 %s12, %c30
  %s14 = or i32 %s13, %c31
  %s15 = or i32 %s14, %c32
  %s16 = or i32 %s15, %c33
  %s17 = or i32 %s16, %c34
  %s18 = or i32 %s17, %c35
  %s19 = or i32 %s18, %c36
  %s20 = or i32 %s19, %c37
  %s21 = or i32 %s20, %c38
  %s22 = or i32 %s21, %c39
  %s23 = or i32 %s22, %c40
  %s24 = or i32 %s23, %c41
  %s25 = or i32 %s24, %c42
  %s26 = or i32 %s25, %c43
  %bf = icmp ne i32 %s26, 0
  br i1 %bf, label %fail_pairs, label %t_vec

t_vec:
  ; Vector <2 x double>: lane 0 = (-0.0, +0.0), lane 1 = (+0.0, -0.0).
  %vx = insertelement <2 x double> <double -0.0, double 0.0>, double 0.0, i32 1
  %vy = insertelement <2 x double> <double 0.0, double -0.0>, double -0.0, i32 1
  %vmax_rev = call { <2 x double>, <2 x double> } (...) @__enzyme_autodiff_v2(ptr @fmax_v2, <2 x double> %vx, <2 x double> %vy)
  %vmax_dx = extractvalue { <2 x double>, <2 x double> } %vmax_rev, 0
  %vmax_dy = extractvalue { <2 x double>, <2 x double> } %vmax_rev, 1
  %vmax_dx0 = extractelement <2 x double> %vmax_dx, i32 0
  %vmax_dx1 = extractelement <2 x double> %vmax_dx, i32 1
  %vmax_dy0 = extractelement <2 x double> %vmax_dy, i32 0
  %vmax_dy1 = extractelement <2 x double> %vmax_dy, i32 1
  %vmx0 = call i32 @check_scalar(double %vmax_dx0, double 0.0, i32 110)
  %vmx1 = call i32 @check_scalar(double %vmax_dx1, double 1.0, i32 111)
  %vmy0 = call i32 @check_scalar(double %vmax_dy0, double 1.0, i32 112)
  %vmy1 = call i32 @check_scalar(double %vmax_dy1, double 0.0, i32 113)

  %vmin_fwd = call <2 x double> (...) @__enzyme_fwddiff_v2(ptr @fmin_v2, <2 x double> %vx, <2 x double> <double 1.0, double 1.0>, <2 x double> %vy, <2 x double> <double 0.0, double 0.0>)
  %vmin_f0 = extractelement <2 x double> %vmin_fwd, i32 0
  %vmin_f1 = extractelement <2 x double> %vmin_fwd, i32 1
  %vnf0 = call i32 @check_scalar(double %vmin_f0, double 1.0, i32 114)
  %vnf1 = call i32 @check_scalar(double %vmin_f1, double 0.0, i32 115)

  %v0 = or i32 %vmx0, %vmx1
  %v1 = or i32 %v0, %vmy0
  %v2 = or i32 %v1, %vmy1
  %v3 = or i32 %v2, %vnf0
  %v4 = or i32 %v3, %vnf1
  ret i32 %v4

fail_pairs:
  ret i32 %s26

fail:
  %ef = phi i32 [ %e1, %entry ], [ %e2, %t2 ], [ %e3, %t3 ], [ %e4, %t4 ], [ %e5, %t5 ], [ %e6, %t6 ], [ %e7, %t7 ], [ %e8, %t8 ]
  ret i32 %ef
}
