; RUN: if [ %llvmver -lt 16 ] && [ %llvmver -ge 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

declare void @__enzyme_reverse(...) 

declare i64 @foo() "enzyme_inactive" "enzyme_nofree" "enzyme_no_escaping_allocation"

declare void @cuStreamSynchronize(i64)

define void @square() {
entry:
  %z = call i64 @foo()
  call void @cuStreamSynchronize(i64 %z)
  ret void
}

define void @dsquare() {
entry:
  tail call void (...) @__enzyme_reverse(void ()* nonnull @square, i8* null)
  ret void
}

; CHECK: define internal void @diffesquare(i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast i8* %tapeArg to i64*
; CHECK-NEXT:   %z = load i64, i64* %0
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   call void @cuStreamSynchronize(i64 %z)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

