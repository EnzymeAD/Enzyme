; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s; fi

; An inactive callee is never differentiated, so its body is not analyzed:
; only its annotations type the call. Here @mid, like a Fortran I/O routine
; of a COMMON block that holds a double array followed by a float array,
; passes the block (from its base) to @r8, whose parameter says the whole
; object is double, and the float array after it to @r4, whose parameter says
; float. Analyzing @mid's body used to abort with "Illegal updateAnalysis".

@blk = internal global [16 x i8] zeroinitializer, align 8

define void @r8(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %p, ptr %q) {
  ret void
}

define void @r4(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float}" %p, ptr %q) {
  ret void
}

define void @mid(ptr %q) "enzyme_inactive" {
  call void @r8(ptr @blk, ptr %q)
  call void @r4(ptr getelementptr inbounds (i8, ptr @blk, i64 8), ptr %q)
  ret void
}

define void @caller(ptr %q) {
entry:
  call void @mid(ptr %q)
  ret void
}

; CHECK: caller - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: ptr %q: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   call void @mid(ptr %q): {}
; CHECK-NEXT:   ret void: {}
