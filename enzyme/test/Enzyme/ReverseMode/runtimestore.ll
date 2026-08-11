; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme-loose-types -enzyme -mem2reg -instsimplify -adce -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme-loose-types -passes="enzyme,function(mem2reg,instsimplify,adce,%simplifycfg,early-cse)" -S | FileCheck %s

declare void @__enzyme_autodiff(...)

define void @f() local_unnamed_addr {
bb:
  call void (...) @__enzyme_autodiff(void (double*, double)* nonnull @julia___2553_inner.1, metadata !"enzyme_runtime_activity", double* null, double* null, double 1.0)
  ret void
}

define void @julia___2553_inner.1(double* %arg, double %arg2) {
bb:
  store double %arg2, double* %arg
  ret void
}

; CHECK: define internal { double } @diffejulia___2553_inner.1(double* %arg, double* %"arg'", double %arg2)
; CHECK-NEXT: bb:
; CHECK-NEXT:   store double %arg2, double* %arg
; CHECK-NEXT:   %[[cmp:.+]] = icmp ne double* %arg, %"arg'"
; CHECK-NEXT:   br i1 %[[cmp]], label %invertbb_active, label %invertbb_amerge

; CHECK: invertbb_active:                                  ; preds = %bb
; CHECK-NEXT:   %[[act:.+]] = load double, double* %"arg'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"arg'"
; CHECK-NEXT:   br label %invertbb_amerge

; CHECK: invertbb_amerge:                                  ; preds = %invertbb_active, %bb
; CHECK-NEXT:   %[[phi:.+]] = phi double [ %[[act]], %invertbb_active ], [ 0.000000e+00, %bb ]
; CHECK-NEXT:   %[[res:.+]] = insertvalue { double } undef, double %[[phi]], 0
; CHECK-NEXT:   ret { double } %[[res]]
; CHECK-NEXT: }
