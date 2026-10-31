; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller_untyped -S -o /dev/null | FileCheck %s --check-prefix=UNTYPED; fi

; An inactive callee whose interface is fully typed is never differentiated,
; so its body is not analyzed: only its annotations type the call. Here @mid,
; like a Fortran I/O routine of a COMMON block that holds a double array
; followed by a float array, passes the block (from its base) to @r8, whose
; parameter says the whole object is double, and the float array after it to
; @r4, whose parameter says float. Analyzing @mid's body used to abort with
; "Illegal updateAnalysis".

@blk = internal global [16 x i8] zeroinitializer, align 8

define void @r8(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@double}" %p, ptr %q) {
  ret void
}

define void @r4(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Float@float}" %p, ptr %q) {
  ret void
}

define void @mid(ptr "enzyme_type"="{[-1]:Pointer}" %q) "enzyme_inactive" {
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

; Without annotations the body of an inactive callee is the only source of the
; types it gives its arguments (here, C++ code setting an integer stride), so
; it is still analyzed.

define void @set_stride(ptr %t) "enzyme_inactive" {
  %s = getelementptr inbounds i8, ptr %t, i64 8
  store i64 1, ptr %s, align 8
  ret void
}

define void @caller_untyped(ptr %t) {
entry:
  call void @set_stride(ptr %t)
  ret void
}

; UNTYPED: caller_untyped - {} |{[-1]:Pointer}:{}
; UNTYPED-NEXT: ptr %t: {[-1]:Pointer, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}
; UNTYPED-NEXT: entry
; UNTYPED-NEXT:   call void @set_stride(ptr %t): {}
; UNTYPED-NEXT:   ret void: {}
