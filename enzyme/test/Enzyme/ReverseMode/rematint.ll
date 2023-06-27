; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define void @f(i64* %arg, {} addrspace(10)* %arg1) {
bb:
  %i7 = alloca i64, align 8
  %i5 = call [1 x i64] @a2(i64* %arg, {} addrspace(10)* %arg1)
  %i6 = extractvalue [1 x i64] %i5, 0
  store i64 %i6, i64* %i7, align 8
  call void @a3(i64* %arg, i64* %i7)
  ret void
}

declare void @__enzyme_reverse(...)

define void @dsquare(double %arg) {
bb:
  call void (...) @__enzyme_reverse(void (i64*, {} addrspace(10)*)* nonnull @f, metadata !"enzyme_dup", i64* undef, i64* undef, metadata !"enzyme_dup", {} addrspace(10)* undef, {} addrspace(10)* undef, i8* null)
  ret void
}

define [1 x i64] @a2(i64* %arg, {} addrspace(10)* %arg1) {
bb:
  %i5 = load i64, i64* %arg, align 8, !tbaa !5
  %i30 = insertvalue [1 x i64] undef, i64 %i5, 0
  ret [1 x i64] %i30
}

define void @a3(i64* %arg, i64* nocapture readonly %arg1) {
bb:
  ret void
}

!5 = !{!6, !6, i64 0}
!6 = !{!"jtbaa_arraylen", !7, i64 0}
!7 = !{!"jtbaa_array", !8, i64 0}
!8 = !{!"jtbaa", !9, i64 0}
!9 = !{!"jtbaa"}

; CHECK: define internal i8* @augmented_f(i64* %arg, i64* %"arg'", {} addrspace(10)* %arg1, {} addrspace(10)* %"arg1'")
; CHECK-NEXT: bb:
; CHECK-NEXT:   %malloccall1 = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   %tapemem = bitcast i8* %malloccall1 to i8**
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
; CHECK-NEXT:   store i8* %malloccall, i8** %tapemem
; CHECK-NEXT:   %i7 = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   %i5 = call [1 x i64] @augmented_a2(i64* %arg, i64* %"arg'", {} addrspace(10)* %arg1, {} addrspace(10)* %"arg1'")
; CHECK-NEXT:   %i6 = extractvalue [1 x i64] %i5, 0
; CHECK-NEXT:   store i64 %i6, i64* %i7, align 8
; CHECK-NEXT:   call void @augmented_a3(i64* %arg, i64* %"arg'", i64* %i7)
; CHECK-NEXT:   ret i8* %malloccall1
; CHECK-NEXT: }

; CHECK: define internal void @diffef(i64* %arg, i64* %"arg'", {} addrspace(10)* %arg1, {} addrspace(10)* %"arg1'", i8* %tapeArg) 
; CHECK-NEXT: bb:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to i8**
; CHECK-NEXT:   %malloccall = load i8*, i8** %0
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %i7 = bitcast i8* %malloccall to i64*
; CHECK-NEXT:   call void @diffea3(i64* %arg, i64* %"arg'", i64* %i7)
; CHECK-NEXT:   call void @diffea2(i64* %arg, i64* %"arg'", {} addrspace(10)* %arg1, {} addrspace(10)* %"arg1'")
; CHECK-NEXT:   call void @free(i8* %malloccall)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
