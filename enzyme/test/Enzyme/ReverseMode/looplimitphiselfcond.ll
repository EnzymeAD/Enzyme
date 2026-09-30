; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL ; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EVAL

; The inner loop's limit %n is a loop-carried phi of the outer loop, whose
; backedge value %nnext is itself a phi selected by branch conditions that
; depend on %n. Recomputing the limit in the reverse pass unrolls %n to its
; previous iteration, and while doing so %n is marked as unavailable to prevent
; a recursive unroll. Unwrapping the branch conditions then asks for %n again,
; which must fail gracefully rather than dereference the unavailable marker.

; EVAL: d0=1.000000 d1=0.000000 d2=1.000000

@fmt = private unnamed_addr constant [19 x i8] c"d0=%f d1=%f d2=%f\0A\00", align 1
@nglob = private global i64 3, align 8

declare i32 @printf(i8*, ...)

declare double @__enzyme_autodiff(i8*, ...)

define double @f(double* %P, double* %x, i64* %np, i64 %m) {
entry:
  %n0 = load i64, i64* %np, align 8
  br label %outer

outer:
  %n = phi i64 [ %n0, %entry ], [ %nnext, %latch ]
  %k = phi i64 [ 0, %entry ], [ %knext, %latch ]
  %s = phi double [ 0.000000e+00, %entry ], [ %snext, %latch ]
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %thread, label %nonempty

thread:
  br label %merge

nonempty:
  %pre = load i64, i64* %np, align 8
  %one = icmp eq i64 %n, 1
  br i1 %one, label %single, label %multi

single:
  br label %merge

multi:
  br label %merge

merge:
  %nnext = phi i64 [ 0, %thread ], [ %pre, %single ], [ %pre, %multi ]
  br i1 %empty, label %latch, label %inner.ph

inner.ph:
  %kf = sitofp i64 %k to double
  br label %inner

inner:
  %j = phi i64 [ 0, %inner.ph ], [ %jnext, %inner ]
  %acc = phi double [ %s, %inner.ph ], [ %accnext, %inner ]
  %xg = getelementptr inbounds double, double* %x, i64 %j
  %xv = load double, double* %xg, align 8
  %cmp = fcmp oeq double %xv, %kf
  %pg = getelementptr inbounds double, double* %P, i64 %j
  %pv = load double, double* %pg, align 8
  %sel = select i1 %cmp, double %pv, double 0.000000e+00
  %accnext = fadd double %acc, %sel
  %jnext = add nuw nsw i64 %j, 1
  %done = icmp eq i64 %jnext, %n
  br i1 %done, label %latch, label %inner

latch:
  %snext = phi double [ %s, %merge ], [ %accnext, %inner ]
  %knext = add nuw nsw i64 %k, 1
  %exit = icmp eq i64 %knext, %m
  br i1 %exit, label %end, label %outer

end:
  ; Overwrite the mask input, so that the select conditions must be cached.
  store double -1.000000e+00, double* %x, align 8
  ret double %snext
}

define i32 @main() {
entry:
  %P = alloca [3 x double], align 8
  %dP = alloca [3 x double], align 8
  %x = alloca [3 x double], align 8
  %P0 = getelementptr inbounds [3 x double], [3 x double]* %P, i64 0, i64 0
  %P1 = getelementptr inbounds [3 x double], [3 x double]* %P, i64 0, i64 1
  %P2 = getelementptr inbounds [3 x double], [3 x double]* %P, i64 0, i64 2
  %dP0 = getelementptr inbounds [3 x double], [3 x double]* %dP, i64 0, i64 0
  %dP1 = getelementptr inbounds [3 x double], [3 x double]* %dP, i64 0, i64 1
  %dP2 = getelementptr inbounds [3 x double], [3 x double]* %dP, i64 0, i64 2
  %x0 = getelementptr inbounds [3 x double], [3 x double]* %x, i64 0, i64 0
  %x1 = getelementptr inbounds [3 x double], [3 x double]* %x, i64 0, i64 1
  %x2 = getelementptr inbounds [3 x double], [3 x double]* %x, i64 0, i64 2
  store double 2.000000e+00, double* %P0, align 8
  store double 3.000000e+00, double* %P1, align 8
  store double 5.000000e+00, double* %P2, align 8
  store double 0.000000e+00, double* %dP0, align 8
  store double 0.000000e+00, double* %dP1, align 8
  store double 0.000000e+00, double* %dP2, align 8
  ; With k in 0..1 the mask selects x == 0 and x == 1, but not x == 7.
  store double 1.000000e+00, double* %x0, align 8
  store double 7.000000e+00, double* %x1, align 8
  store double 0.000000e+00, double* %x2, align 8
  %r = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, double*, i64*, i64)* @f to i8*), metadata !"enzyme_dup", double* %P0, double* %dP0, metadata !"enzyme_const", double* %x0, metadata !"enzyme_const", i64* @nglob, i64 2)
  %d0 = load double, double* %dP0, align 8
  %d1 = load double, double* %dP1, align 8
  %d2 = load double, double* %dP2, align 8
  %p = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @fmt, i64 0, i64 0), double %d0, double %d1, double %d2)
  ret i32 0
}
