; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=0 -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=0 -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=0 -passes="enzyme,default<O2>" -S | FileCheck %s --check-prefix=OPT
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=0 -passes="enzyme,default<O2>" | %lli -

declare void @abort()
declare void @__enzyme_autodiff(...)
declare double @__enzyme_autodiff_val(...)

@enzyme_const = external global i32
@enzyme_dup = external global i32

define double @guarded_noundef(double noundef %x) #0 {
entry:
  ret double %x
}

define double @nested_staging_noundef(ptr nocapture readonly %fvec, ptr nocapture %canary, i1 zeroext %fan) #1 {
entry:
  %scratch = alloca [2 x double], align 16
  br i1 %fan, label %fan.entry, label %merge

fan.entry:
  store i32 1, ptr %canary, align 4
  %init = load i64, ptr %fvec, align 8
  store i64 %init, ptr %scratch, align 16
  br label %sum.loop

sum.loop:
  %iv = phi i64 [ 0, %fan.entry ], [ %iv.next, %sum.loop ]
  %acc = phi double [ 0.000000e+00, %fan.entry ], [ %acc.next, %sum.loop ]
  %gep = getelementptr inbounds double, ptr %fvec, i64 %iv
  %val = load double, ptr %gep, align 8
  %acc.next = fadd double %acc, %val
  %iv.next = add nuw nsw i64 %iv, 1
  %cont = icmp eq i64 %iv, 0
  br i1 %cont, label %sum.loop, label %check.rad

check.rad:
  %rad = call double @guarded_noundef(double noundef %acc.next)
  %pos = fcmp ogt double %rad, 0.000000e+00
  br i1 %pos, label %scale.loop, label %merge

scale.loop:
  %iv2 = phi i64 [ 0, %check.rad ], [ %iv2.next, %scale.loop ]
  %sgep = getelementptr inbounds [2 x double], ptr %scratch, i64 0, i64 %iv2
  %sval = load double, ptr %sgep, align 8
  %sdiv = fdiv double %sval, %rad
  store double %sdiv, ptr %sgep, align 8
  %iv2.next = add nuw nsw i64 %iv2, 1
  %cont2 = icmp eq i64 %iv2, 0
  br i1 %cont2, label %scale.loop, label %merge

merge:
  %out = load double, ptr %fvec, align 8
  ret double %out
}

define double @nested_active_phi(double %x, ptr nocapture readonly %maybe_null, i1 zeroext %fan) #1 {
entry:
  br i1 %fan, label %inner, label %merge

inner:
  %loaded = load double, ptr %maybe_null, align 8
  %pos = fcmp ogt double %loaded, 0.000000e+00
  br i1 %pos, label %pos.bb, label %neg.bb

pos.bb:
  %mul2 = fmul double %x, 2.000000e+00
  br label %merge

neg.bb:
  %mul3 = fmul double %x, 3.000000e+00
  br label %merge

merge:
  %res = phi double [ %x, %entry ], [ %mul2, %pos.bb ], [ %mul3, %neg.bb ]
  ret double %res
}

define double @nested_unwrap_phi(double %x, ptr nocapture readonly %weights, ptr nocapture readonly %maybe_null, i1 zeroext %run, i1 zeroext %fan) #1 {
entry:
  br i1 %run, label %guard, label %fin

guard:
  br i1 %fan, label %inner, label %merge

inner:
  %loaded = load double, ptr %maybe_null, align 8
  %pos = fcmp ogt double %loaded, 0.000000e+00
  br i1 %pos, label %pos.bb, label %neg.bb

pos.bb:
  %w1_ptr = getelementptr inbounds double, ptr %weights, i64 1
  %w1 = load double, ptr %w1_ptr, align 8
  br label %merge

neg.bb:
  %w2_ptr = getelementptr inbounds double, ptr %weights, i64 2
  %w2 = load double, ptr %w2_ptr, align 8
  br label %merge

merge:
  %scale = phi double [ 1.000000e+00, %guard ], [ %w1, %pos.bb ], [ %w2, %neg.bb ]
  %prod = fmul double %x, %scale
  br label %fin

fin:
  %total = phi double [ 0.000000e+00, %entry ], [ %prod, %merge ]
  ret double %total
}

define void @check_eq(double %got, double %expected) {
entry:
  %diff = fsub double %got, %expected
  %neg = fsub double 0.000000e+00, %diff
  %isneg = fcmp olt double %diff, 0.000000e+00
  %abs = select i1 %isneg, double %neg, double %diff
  %bad = fcmp ogt double %abs, 1.000000e-09
  br i1 %bad, label %fail, label %ok

fail:
  call void @abort()
  unreachable

ok:
  ret void
}

define i32 @main() {
entry:
  %fvec = alloca [2 x double], align 16
  %dfvec = alloca [2 x double], align 16
  %weights = alloca [3 x double], align 16
  %canary = alloca i32, align 4
  %pos_val = alloca double, align 8
  %neg_val = alloca double, align 8

  %fvec1 = getelementptr inbounds [2 x double], ptr %fvec, i64 0, i64 1
  %dfvec1 = getelementptr inbounds [2 x double], ptr %dfvec, i64 0, i64 1
  store double 1.000000e+00, ptr %fvec, align 16
  store double 2.000000e+00, ptr %fvec1, align 8
  store double 0.000000e+00, ptr %dfvec, align 16
  store double 0.000000e+00, ptr %dfvec1, align 8
  store i32 0, ptr %canary, align 4

  %w1p = getelementptr inbounds [3 x double], ptr %weights, i64 0, i64 1
  %w2p = getelementptr inbounds [3 x double], ptr %weights, i64 0, i64 2
  store double 1.000000e+00, ptr %weights, align 16
  store double 2.000000e+00, ptr %w1p, align 8
  store double 3.000000e+00, ptr %w2p, align 16

  %dup = load i32, ptr @enzyme_dup, align 4
  %cst = load i32, ptr @enzyme_const, align 4

  call void (...) @__enzyme_autodiff(ptr @nested_staging_noundef, i32 %dup, ptr %fvec, ptr %dfvec, i32 %cst, ptr %canary, i32 %cst, i1 false)
  %c0 = load i32, ptr %canary, align 4
  %c0_bad = icmp ne i32 %c0, 0
  br i1 %c0_bad, label %fail, label %cont1

cont1:
  %d0 = load double, ptr %dfvec, align 16
  call void @check_eq(double %d0, double 1.000000e+00)

  store double 0.000000e+00, ptr %dfvec, align 16
  call void (...) @__enzyme_autodiff(ptr @nested_staging_noundef, i32 %dup, ptr %fvec, ptr %dfvec, i32 %cst, ptr %canary, i32 %cst, i1 true)
  %c1 = load i32, ptr %canary, align 4
  %c1_bad = icmp ne i32 %c1, 1
  br i1 %c1_bad, label %fail, label %cont2

cont2:
  %d1 = load double, ptr %dfvec, align 16
  call void @check_eq(double %d1, double 1.000000e+00)

  store double 4.000000e+00, ptr %pos_val, align 8
  store double -4.000000e+00, ptr %neg_val, align 8

  %ap_false = call double (...) @__enzyme_autodiff_val(ptr @nested_active_phi, double 5.000000e+00, i32 %cst, ptr null, i32 %cst, i1 false)
  call void @check_eq(double %ap_false, double 1.000000e+00)
  %ap_pos = call double (...) @__enzyme_autodiff_val(ptr @nested_active_phi, double 5.000000e+00, i32 %cst, ptr %pos_val, i32 %cst, i1 true)
  call void @check_eq(double %ap_pos, double 2.000000e+00)
  %ap_neg = call double (...) @__enzyme_autodiff_val(ptr @nested_active_phi, double 5.000000e+00, i32 %cst, ptr %neg_val, i32 %cst, i1 true)
  call void @check_eq(double %ap_neg, double 3.000000e+00)

  %up_false = call double (...) @__enzyme_autodiff_val(ptr @nested_unwrap_phi, double 5.000000e+00, i32 %cst, ptr %weights, i32 %cst, ptr null, i32 %cst, i1 true, i32 %cst, i1 false)
  call void @check_eq(double %up_false, double 1.000000e+00)
  %up_pos = call double (...) @__enzyme_autodiff_val(ptr @nested_unwrap_phi, double 5.000000e+00, i32 %cst, ptr %weights, i32 %cst, ptr %pos_val, i32 %cst, i1 true, i32 %cst, i1 true)
  call void @check_eq(double %up_pos, double 2.000000e+00)
  %up_neg = call double (...) @__enzyme_autodiff_val(ptr @nested_unwrap_phi, double 5.000000e+00, i32 %cst, ptr %weights, i32 %cst, ptr %neg_val, i32 %cst, i1 true, i32 %cst, i1 true)
  call void @check_eq(double %up_neg, double 3.000000e+00)

  ret i32 0

fail:
  call void @abort()
  unreachable
}

attributes #0 = { noinline norecurse nounwind readnone willreturn }
attributes #1 = { noinline norecurse nounwind }

; CHECK-LABEL: define internal void @diffenested_staging_noundef(
; CHECK: invertmerge:
; CHECK-NOT: call double @guarded_noundef
; CHECK:   br i1 %fan, label %staging, label %invertentry
; CHECK: staging:
; CHECK-NEXT:   %[[CACHED:.+]] = load double, ptr %"acc.next!manual_lcssa_cache"
; CHECK-NEXT:   %[[RAD:.+]] = call double @guarded_noundef(double noundef %[[CACHED]])
; CHECK-NEXT:   %[[CMP:.+]] = fcmp ogt double %[[RAD]], 0.000000e+00
; CHECK-NEXT:   br i1 %[[CMP]], label %{{.+}}, label %invertcheck.rad

; CHECK-LABEL: define internal { double } @diffenested_active_phi(
; CHECK: invertmerge:
; CHECK-NOT: load double, ptr %maybe_null
; CHECK:   %[[SEL:.+]] = load i8, ptr %_cache
; CHECK:   br i1 %fan, label %staging, label %invertentry
; CHECK: staging:
; CHECK-NEXT:   %[[LD1:.+]] = load double, ptr %maybe_null
; CHECK-NEXT:   %[[POS1:.+]] = fcmp ogt double %[[LD1]], 0.000000e+00
; CHECK-NEXT:   br i1 %[[POS1]], label %invertpos.bb, label %invertneg.bb

; CHECK-LABEL: define internal { double } @diffenested_unwrap_phi(
; CHECK: invertmerge:
; CHECK-NOT: load double, ptr %maybe_null
; CHECK:   br i1 %fan, label %invertmerge_phisplt, label %{{.+}}
; CHECK: invertmerge_phisplt:
; CHECK-NEXT:   %[[LD2:.+]] = load double, ptr %maybe_null
; CHECK-NEXT:   %[[POS2:.+]] = fcmp ogt double %[[LD2]], 0.000000e+00
; CHECK-NEXT:   br i1 %[[POS2]],

; OPT-LABEL: define internal fastcc void @diffenested_staging_noundef(
; OPT-NOT: call void @llvm.assume
; OPT: ret void
