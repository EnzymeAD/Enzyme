; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

@.fmt = private unnamed_addr constant [37 x i8] c"t1=%f,%f t2=%f,%f t3=%f,%f t4=%f,%f\0A\00", align 1

declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

define void @use(ptr %p) noinline "enzyme_inactive" nofree {
entry:
  ret void
}

; Active { ptr, i64 } PHI whose i64 lane is constant 0 (BaseType::Anything).
; TypeResults::anyFloat(..., /*anythingIsFloat=*/false) must treat Anything as
; non-float rather than misclassifying the PHI as containing floats.
define double @slice_anything_phi(i1 %cond, ptr %p1, ptr %p2, ptr %sink) {
entry:
  %s1 = insertvalue { ptr, i64 } undef, ptr %p1, 0
  %s1b = insertvalue { ptr, i64 } %s1, i64 0, 1
  %s2 = insertvalue { ptr, i64 } undef, ptr %p2, 0
  %s2b = insertvalue { ptr, i64 } %s2, i64 0, 1
  br i1 %cond, label %then, label %else
then:
  call void @use(ptr %p1)
  br label %merge
else:
  call void @use(ptr %p2)
  br label %merge
merge:
  %phi = phi { ptr, i64 } [ %s1b, %then ], [ %s2b, %else ]
  store { ptr, i64 } %phi, ptr %sink, align 8
  %ptr = extractvalue { ptr, i64 } %phi, 0
  %v = load double, ptr %ptr, align 8
  %sq = fmul double %v, %v
  ret double %sq
}

; Active { i8, { i8, ptr } } PHI loaded from memory.
; skippedBytes must recurse into nested StructTypes using offset + off so
; padding bytes in the inner struct (offsets 9..15) are skipped.
define double @nested_struct_pad_phi(i1 %cond, ptr %in1, ptr %in2, ptr %sink) {
entry:
  br i1 %cond, label %then, label %else
then:
  %s1 = load { i8, { i8, ptr } }, ptr %in1, align 8
  call void @use(ptr %in1)
  br label %merge
else:
  %s2 = load { i8, { i8, ptr } }, ptr %in2, align 8
  call void @use(ptr %in2)
  br label %merge
merge:
  %phi = phi { i8, { i8, ptr } } [ %s1, %then ], [ %s2, %else ]
  store { i8, { i8, ptr } } %phi, ptr %sink, align 8
  %a0 = extractvalue { i8, { i8, ptr } } %phi, 0
  %inner = extractvalue { i8, { i8, ptr } } %phi, 1
  %a1 = extractvalue { i8, ptr } %inner, 0
  %sum = add i8 %a0, %a1
  %ptr = extractvalue { i8, ptr } %inner, 1
  %v = load double, ptr %ptr, align 8
  %sq = fmul double %v, %v
  ret double %sq
}

; Active [2 x { i8, ptr }] PHI loaded from memory.
; skippedBytes must recurse into ArrayTypes so struct padding inside each array
; element is skipped.
define double @array_struct_pad_phi(i1 %cond, ptr %in1, ptr %in2, ptr %sink) {
entry:
  br i1 %cond, label %then, label %else
then:
  %s1 = load [2 x { i8, ptr }], ptr %in1, align 8
  call void @use(ptr %in1)
  br label %merge
else:
  %s2 = load [2 x { i8, ptr }], ptr %in2, align 8
  call void @use(ptr %in2)
  br label %merge
merge:
  %phi = phi [2 x { i8, ptr }] [ %s1, %then ], [ %s2, %else ]
  store [2 x { i8, ptr }] %phi, ptr %sink, align 8
  %e0 = extractvalue [2 x { i8, ptr }] %phi, 0
  %e1 = extractvalue [2 x { i8, ptr }] %phi, 1
  %a0 = extractvalue { i8, ptr } %e0, 0
  %a1 = extractvalue { i8, ptr } %e1, 0
  %sum = add i8 %a0, %a1
  %p0 = extractvalue { i8, ptr } %e0, 1
  %p1 = extractvalue { i8, ptr } %e1, 1
  %v0 = load double, ptr %p0, align 8
  %v1 = load double, ptr %p1, align 8
  %sq0 = fmul double %v0, %v0
  %t1 = fmul double %v1, 2.000000e+00
  %res = fadd double %sq0, %t1
  ret double %res
}

; Active { double, i8 } PHI with constant 0 i8 (BaseType::Anything) and an
; active double field: anyFloat must still return true and differentiate double.
define double @float_anything_phi(i1 %cond, double %x, double %y, ptr %sink) {
entry:
  %x2 = fmul double %x, %x
  %y3 = fmul double %y, 3.000000e+00
  %s1 = insertvalue { double, i8 } undef, double %x2, 0
  %s1b = insertvalue { double, i8 } %s1, i8 0, 1
  %s2 = insertvalue { double, i8 } undef, double %y3, 0
  %s2b = insertvalue { double, i8 } %s2, i8 0, 1
  br i1 %cond, label %then, label %else
then:
  call void @use(ptr %sink)
  br label %merge
else:
  call void @use(ptr %sink)
  br label %merge
merge:
  %phi = phi { double, i8 } [ %s1b, %then ], [ %s2b, %else ]
  store { double, i8 } %phi, ptr %sink, align 8
  %v = extractvalue { double, i8 } %phi, 0
  ret double %v
}

define i32 @main() {
entry:
  ; --- Test 1: slice_anything_phi (cond = true) ---
  %p1 = alloca double, align 8
  %dp1 = alloca double, align 8
  %p2 = alloca double, align 8
  %dp2 = alloca double, align 8
  %sink1 = alloca { ptr, i64 }, align 8
  %dsink1 = alloca { ptr, i64 }, align 8
  store double 3.000000e+00, ptr %p1, align 8
  store double 0.000000e+00, ptr %dp1, align 8
  store double 5.000000e+00, ptr %p2, align 8
  store double 0.000000e+00, ptr %dp2, align 8
  store { ptr, i64 } zeroinitializer, ptr %sink1, align 8
  store { ptr, i64 } zeroinitializer, ptr %dsink1, align 8
  call void (...) @__enzyme_autodiff(ptr @slice_anything_phi, i1 true, ptr %p1, ptr %dp1, ptr %p2, ptr %dp2, ptr %sink1, ptr %dsink1)
  %g1a = load double, ptr %dp1, align 8
  %g1b = load double, ptr %dp2, align 8

  ; --- Test 2: nested_struct_pad_phi (cond = false) ---
  %n1 = alloca { i8, { i8, ptr } }, align 8
  %dn1 = alloca { i8, { i8, ptr } }, align 8
  %n2 = alloca { i8, { i8, ptr } }, align 8
  %dn2 = alloca { i8, { i8, ptr } }, align 8
  %sink2 = alloca { i8, { i8, ptr } }, align 8
  %dsink2 = alloca { i8, { i8, ptr } }, align 8
  store double 0.000000e+00, ptr %dp1, align 8
  store double 0.000000e+00, ptr %dp2, align 8
  %ni1_0 = insertvalue { i8, ptr } undef, i8 1, 0
  %ni1_1 = insertvalue { i8, ptr } %ni1_0, ptr %p1, 1
  %ns1_0 = insertvalue { i8, { i8, ptr } } undef, i8 2, 0
  %ns1_1 = insertvalue { i8, { i8, ptr } } %ns1_0, { i8, ptr } %ni1_1, 1
  store { i8, { i8, ptr } } %ns1_1, ptr %n1, align 8
  %dni1_0 = insertvalue { i8, ptr } undef, i8 0, 0
  %dni1_1 = insertvalue { i8, ptr } %dni1_0, ptr %dp1, 1
  %dns1_0 = insertvalue { i8, { i8, ptr } } undef, i8 0, 0
  %dns1_1 = insertvalue { i8, { i8, ptr } } %dns1_0, { i8, ptr } %dni1_1, 1
  store { i8, { i8, ptr } } %dns1_1, ptr %dn1, align 8
  %ni2_0 = insertvalue { i8, ptr } undef, i8 3, 0
  %ni2_1 = insertvalue { i8, ptr } %ni2_0, ptr %p2, 1
  %ns2_0 = insertvalue { i8, { i8, ptr } } undef, i8 4, 0
  %ns2_1 = insertvalue { i8, { i8, ptr } } %ns2_0, { i8, ptr } %ni2_1, 1
  store { i8, { i8, ptr } } %ns2_1, ptr %n2, align 8
  %dni2_0 = insertvalue { i8, ptr } undef, i8 0, 0
  %dni2_1 = insertvalue { i8, ptr } %dni2_0, ptr %dp2, 1
  %dns2_0 = insertvalue { i8, { i8, ptr } } undef, i8 0, 0
  %dns2_1 = insertvalue { i8, { i8, ptr } } %dns2_0, { i8, ptr } %dni2_1, 1
  store { i8, { i8, ptr } } %dns2_1, ptr %dn2, align 8
  store { i8, { i8, ptr } } zeroinitializer, ptr %sink2, align 8
  store { i8, { i8, ptr } } zeroinitializer, ptr %dsink2, align 8
  call void (...) @__enzyme_autodiff(ptr @nested_struct_pad_phi, i1 false, ptr %n1, ptr %dn1, ptr %n2, ptr %dn2, ptr %sink2, ptr %dsink2)
  %g2a = load double, ptr %dp1, align 8
  %g2b = load double, ptr %dp2, align 8

  ; --- Test 3: array_struct_pad_phi (cond = true) ---
  %a1 = alloca [2 x { i8, ptr }], align 8
  %da1 = alloca [2 x { i8, ptr }], align 8
  %a2 = alloca [2 x { i8, ptr }], align 8
  %da2 = alloca [2 x { i8, ptr }], align 8
  %sink3 = alloca [2 x { i8, ptr }], align 8
  %dsink3 = alloca [2 x { i8, ptr }], align 8
  store double 0.000000e+00, ptr %dp1, align 8
  store double 0.000000e+00, ptr %dp2, align 8
  %ae0 = insertvalue { i8, ptr } { i8 1, ptr null }, ptr %p1, 1
  %ae1 = insertvalue { i8, ptr } { i8 2, ptr null }, ptr %p2, 1
  %av0 = insertvalue [2 x { i8, ptr }] undef, { i8, ptr } %ae0, 0
  %av1 = insertvalue [2 x { i8, ptr }] %av0, { i8, ptr } %ae1, 1
  store [2 x { i8, ptr }] %av1, ptr %a1, align 8
  store [2 x { i8, ptr }] %av1, ptr %a2, align 8
  %dae0 = insertvalue { i8, ptr } { i8 0, ptr null }, ptr %dp1, 1
  %dae1 = insertvalue { i8, ptr } { i8 0, ptr null }, ptr %dp2, 1
  %dav0 = insertvalue [2 x { i8, ptr }] undef, { i8, ptr } %dae0, 0
  %dav1 = insertvalue [2 x { i8, ptr }] %dav0, { i8, ptr } %dae1, 1
  store [2 x { i8, ptr }] %dav1, ptr %da1, align 8
  store [2 x { i8, ptr }] %dav1, ptr %da2, align 8
  store [2 x { i8, ptr }] zeroinitializer, ptr %sink3, align 8
  store [2 x { i8, ptr }] zeroinitializer, ptr %dsink3, align 8
  call void (...) @__enzyme_autodiff(ptr @array_struct_pad_phi, i1 true, ptr %a1, ptr %da1, ptr %a2, ptr %da2, ptr %sink3, ptr %dsink3)
  %g3a = load double, ptr %dp1, align 8
  %g3b = load double, ptr %dp2, align 8

  ; --- Test 4: float_anything_phi (cond = true) ---
  %sink4 = alloca { double, i8 }, align 8
  %dsink4 = alloca { double, i8 }, align 8
  store { double, i8 } zeroinitializer, ptr %sink4, align 8
  store { double, i8 } zeroinitializer, ptr %dsink4, align 8
  %g4 = call { double, double } (...) @__enzyme_autodiff(ptr @float_anything_phi, i1 true, double 4.000000e+00, double 5.000000e+00, ptr %sink4, ptr %dsink4)
  %g4a = extractvalue { double, double } %g4, 0
  %g4b = extractvalue { double, double } %g4, 1

  call i32 (ptr, ...) @printf(ptr @.fmt, double %g1a, double %g1b, double %g2a, double %g2b, double %g3a, double %g3b, double %g4a, double %g4b)
  ret i32 0
}

; EVAL: t1=6.000000,0.000000 t2=0.000000,10.000000 t3=6.000000,2.000000 t4=8.000000,0.000000

; CHECK-LABEL: define internal void @diffeslice_anything_phi(
; CHECK: phi { ptr, i64 } [ %"s1b'ipiv", %then ], [ %"s2b'ipiv", %else ]
; CHECK: ret void

; CHECK-LABEL: define internal void @diffenested_struct_pad_phi(
; CHECK: phi { i8, { i8, ptr } } [ %"s1'ipl", %then ], [ %"s2'ipl", %else ]
; CHECK: ret void

; CHECK-LABEL: define internal void @diffearray_struct_pad_phi(
; CHECK: phi [2 x { i8, ptr }] [ %"s1'ipl", %then ], [ %"s2'ipl", %else ]
; CHECK: ret void

; CHECK-LABEL: define internal { double, double } @diffefloat_anything_phi(
; CHECK: %phi = phi { double, i8 } [ %s1b, %then ], [ %s2b, %else ]
; CHECK: ret { double, double }
