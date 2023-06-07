; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: mustprogress uwtable willreturn
define internal void @jac_rev(double* nocapture readonly %r, { i8* } %tapeArg, i1 %cmp) {
entry:
  %arg0 = extractvalue { i8* } %tapeArg, 0
  store i8 0, i8* %arg0, align 8
  br i1 %cmp, label %invertbaz, label %invertfoo

invertbaz:                                        ; preds = %invertfoo, %entry
  %call219pre-phi = phi i8* [ %arg0, %entry ], [ %arg1, %invertfoo ]
  tail call void @free(i8* nonnull %call219pre-phi)
  ret void

invertfoo:                                        ; preds = %entry
  %arg1 = extractvalue { i8* } %tapeArg, 0
  %.preipl_unwrap = load double, double* %r, align 8
  br label %invertbaz
}

define void @hessian(double* %r, { i8* } %tapeArg, { i8* } %dtapeArg) local_unnamed_addr {
entry:
  tail call void (...) @_Z17__enzyme_fwddiff(void (double*, { i8* }, i1)* @jac_rev, metadata !"enzyme_const", double* %r, metadata !"enzyme_dup", { i8* } %tapeArg, { i8* } %dtapeArg, i1 true)
  ret void
}

declare void @_Z17__enzyme_fwddiff(...) 

declare void @free(i8*)

; CHECK: define internal void @fwddiffejac_rev(double* nocapture readonly %r, { i8* } %tapeArg, { i8* } %"tapeArg'", i1 %cmp)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arg0'ipev" = extractvalue { i8* } %"tapeArg'", 0
; CHECK-NEXT:   %arg0 = extractvalue { i8* } %tapeArg, 0
; CHECK-NEXT:   store i8 0, i8* %"arg0'ipev", align 8
; CHECK-NEXT:   store i8 0, i8* %arg0, align 8
; CHECK-NEXT:   br i1 %cmp, label %invertbaz, label %invertfoo

; CHECK: invertbaz:                                        ; preds = %invertfoo, %entry
; CHECK-NEXT:   %0 = phi i8* [ %"arg0'ipev", %entry ], [ %"arg1'ipev", %invertfoo ]
; CHECK-NEXT:   %call219pre-phi = phi i8* [ %arg0, %entry ], [ %arg1, %invertfoo ]
; CHECK-NEXT:   tail call void @free(i8* nonnull %call219pre-phi)
; CHECK-NEXT:   %1 = icmp ne i8* %call219pre-phi, %0
; CHECK-NEXT:   br i1 %1, label %free0.i, label %__enzyme_checked_free_1.exit

; CHECK: free0.i:                                          ; preds = %invertbaz
; CHECK-NEXT:   call void @free(i8* nonnull %0)
; CHECK-NEXT:   br label %__enzyme_checked_free_1.exit

; CHECK: __enzyme_checked_free_1.exit:                     ; preds = %invertbaz, %free0.i
; CHECK-NEXT:   ret void

; CHECK: invertfoo:                                        ; preds = %entry
; CHECK-NEXT:   %"arg1'ipev" = extractvalue { i8* } %"tapeArg'", 0
; CHECK-NEXT:   %arg1 = extractvalue { i8* } %tapeArg, 0
; CHECK-NEXT:   br label %invertbaz
; CHECK-NEXT: }
