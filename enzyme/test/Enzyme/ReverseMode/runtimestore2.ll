; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-loose-types -enzyme -mem2reg -instsimplify -adce -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-loose-types -passes="enzyme,function(mem2reg,instsimplify,adce,%simplifycfg,early-cse)" -S | FileCheck %s

declare { } @__enzyme_reverse(...)

define void @f() local_unnamed_addr {
bb:
  %b = call { } (...) @__enzyme_reverse(void (double*, double)* nonnull @julia___2553_inner.1, metadata !"enzyme_runtime_activity", metadata !"enzyme_dup", double* null, double* null, metadata !"enzyme_const", double 1.0, i8* null)
  ret void
}

define void @julia___2553_inner.1(double* %arg, double %arg2) {
bb:
  %g = getelementptr double, double* %arg, i32 1
  store double %arg2, double* %g
  ret void
}

; CHECK: define internal void @diffejulia___2553_inner.1(double* %arg, double* %"arg'", double %arg2, i8* %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %"g'ipg" = getelementptr double, double* %"arg'", i32 1
; CHECK-NEXT:   %g = getelementptr double, double* %arg, i32 1
; CHECK-NEXT:   %0 = icmp ne double* %g, %"g'ipg"
; CHECK-NEXT:   br i1 %0, label %invertbb_active, label %invertbb_amerge

; CHECK: invertbb_active:                                  ; preds = %bb
; CHECK-NEXT:   store double 0.000000e+00, double* %"g'ipg"
; CHECK-NEXT:   br label %invertbb_amerge

; CHECK: invertbb_amerge:                                  ; preds = %invertbb_active, %bb
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
