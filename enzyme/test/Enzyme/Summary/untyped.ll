; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -disable-output | FileCheck %s; fi

; enzyme-summary: untyped copies (memcpy, memset, flang runtime calls) only
; count as floating-point effects on memory that may hold floating-point
; data. Here a COMMON block annotated as integers, a local SAVE character
; table and an integer dummy are written untyped; only the untyped real
; COMMON block counts.

@chars_ = common global [8 x i8] zeroinitializer, !enzyme_type !0
@reals_ = common global [8 x i8] zeroinitializer
@_QFfEtable = internal global [4 x i8] c"ABCD"

declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
declare i64 @_FortranAIndex1(ptr, i64, ptr, i64, i1)

define void @f(ptr "enzyme_type"="{[-1]:Pointer, [-1,-1]:Integer}" %s) {
  call void @llvm.memset.p0.i64(ptr @chars_, i8 32, i64 8, i1 false)
  call void @llvm.memset.p0.i64(ptr @reals_, i8 0, i64 8, i1 false)
  call void @llvm.memset.p0.i64(ptr %s, i8 32, i64 4, i1 false)
  %i = call i64 @_FortranAIndex1(ptr @_QFfEtable, i64 4, ptr %s, i64 1, i1 false)
  ret void
}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Pointer", i32 -1, !2}
!2 = !{!"Integer"}

; CHECK:      "globals_write": [
; CHECK-NEXT:   "reals_"
; CHECK-NEXT: ]
; CHECK-NOT: "write": true
