; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -gvn -adce -S | FileCheck %s

define void @set(double* nocapture %a, double %x) {
entry:
  store double %x, double* %a, align 8
  ret void
}

define double @above(double %i10) {
entry:
  %m = alloca double, align 8
  call void @set(double* nonnull %m, double %i10)
  %i12 = load double, double* %m, align 8
  ret double %i12
}

define double @msg(double %in) {
entry:
  %hst = call double @above(double %in)
  %r = fmul double %hst, %hst
  ret double %r
}

; Function Attrs: norecurse nounwind uwtable

define double @caller() {
entry:
  %r = call double (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double)* @msg to i8*), double 2.000000e+00)
  ret double %r
}

declare dso_local double @__enzyme_autodiff(i8*, ...)

; CHECK: define internal { double } @diffemsg(double %in, double %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %hst_augmented = call { i8*, double } @augmented_above(double %in)
; CHECK-NEXT:   %subcache = extractvalue { i8*, double } %hst_augmented, 0
; CHECK-NEXT:   %hst = extractvalue { i8*, double } %hst_augmented, 1
; CHECK-NEXT:   %0 = fadd fast double %hst, %hst
; CHECK-NEXT:   %1 = fmul fast double %0, %differeturn
; CHECK-NEXT:   %2 = call { double } @diffeabove(double %in, double %1, i8* %subcache)
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }

; CHECK: define internal void @augmented_set(double* nocapture %a, double* nocapture %"a'", double %x) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   store double %x, double* %a, align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { i8*, double } @augmented_above(double %i10) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias dereferenceable_or_null(8) i8* @malloc(i64 8), !enzyme_fromstack !0
; CHECK-NEXT:   %m = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @augmented_set(double* %m, double* undef, double %i10)
; CHECK-NEXT:   %i12 = load double, double* %m, align 8
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i8*, double } undef, i8* %malloccall, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i8*, double } %.fca.0.insert, double %i12, 1
; CHECK-NEXT:   ret { i8*, double } %.fca.1.insert
; CHECK-NEXT: }
 
; TODO not need to cache the primal
; CHECK: define internal { double } @diffeabove(double %i10, double %differeturn, i8* %malloccall) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"malloccall'mi1" = alloca double, align 8
; CHECK-NEXT:   %tmpcast = bitcast double* %"malloccall'mi1" to i64*
; CHECK-NEXT:   store i64 0, i64* %tmpcast, align 8
; CHECK-NEXT:   %m = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store double %differeturn, double* %"malloccall'mi1", align 8
; CHECK-NEXT:   %0 = call { double } @diffeset(double* %m, double* nonnull %"malloccall'mi1", double %i10)
; CHECK-NEXT:   tail call void @free(i8* %malloccall)
; CHECK-NEXT:   ret { double } %0
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeset(double* nocapture %a, double* nocapture %"a'", double %x) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load double, double* %"a'", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a'", align 8
; CHECK-NEXT:   %1 = insertvalue { double } undef, double %0, 0
; CHECK-NEXT:   ret { double } %1
; CHECK-NEXT: }
