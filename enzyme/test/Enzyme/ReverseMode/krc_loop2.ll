; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define void @f(i64 %i5, i64* noalias %i4, float* noalias %i13, float* noalias %i12)  {
bb:
  %i6 = icmp ult i64 %i5, 2
  br i1 %i6, label %bb48, label %bb7

bb7:                                              ; preds = %bb
  %i8 = load i64, i64* %i4, align 8
  %i9 = add i64 %i8, -1
  br label %bb14

bb14:                                             ; preds = %bb41, %bb7
  %i15 = phi i64 [ %i42, %bb41 ], [ 0, %bb7 ]
  %i45 = icmp eq i64 %i15, %i5
  %i42 = add nuw i64 %i15, 1
  br label %bb17

bb17:                                             ; preds = %bb39, %bb14
  %i18 = phi i64 [ 0, %bb14 ], [ %i19, %bb39 ]
  %i19 = add nuw nsw i64 %i18, 1
  %i40 = icmp eq i64 %i18, %i9
  br label %bb21

bb21:                                             ; preds = %bb32, %bb17
  %i22 = phi i64 [ %i33, %bb32 ], [ 0, %bb17 ]
  %i33 = add i64 %i22, 1
  %i34 = icmp sle i64 %i15, %i22
  br label %bb23

bb23:                                             ; preds = %bb23, %bb21
  %i24 = phi i64 [ %i25, %bb23 ], [ 0, %bb21 ]
  %i25 = add i64 %i24, 1
  %i26 = load float, float* %i12, align 4
  %i27 = icmp sgt i64 %i18, %i24
  br i1 %i27, label %bb32, label %bb23

bb32:                                             ; preds = %bb23
  br i1 %i34, label %bb21, label %bb39

bb39:                                             ; preds = %bb32
  store float %i26, float* %i13, align 4
  br i1 %i40, label %bb41, label %bb17

bb41:                                             ; preds = %bb39
  br i1 %i45, label %bb48, label %bb14

bb48:                                             ; preds = %bb41, %bb
  ret void
}

declare i8* @__enzyme_reverse(...)

define void @main() {
bb:
  %i = call i8* (...) @__enzyme_reverse(void (i64, i64*, float*, float*)* @f, i64 0, i64* null, metadata !"enzyme_dup", float* null, float* null, metadata !"enzyme_dup", float* null, float* null, i8* null)
  ret void
}

; CHECK: define internal void @diffef(i64 %i5, i64* noalias %i4, float* noalias %i13, float* %"i13'", float* noalias %i12, float* %"i12'", i8* %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to i64*
; CHECK-NEXT:   %i9 = load i64, i64* %0, align 4, !enzyme_mustcache
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
