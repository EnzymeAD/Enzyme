; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

; real(8) :: u(0:2, 0:1) in a module. Only the copy of a row tells anything
; about the type of the destination.
@_QMvarsEu = internal global [2 x [3 x double]] zeroinitializer, align 64

; real(8) :: a(4); integer(8) :: b(4); equivalence (a, b). The storage of
; an EQUIVALENCE group is a byte array, whose type is unknown.
@_QMmEa = internal global [32 x i8] zeroinitializer, align 64

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

define void @callee(i8* %row, i8* %bytes) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %row, i8* getelementptr inbounds (i8, i8* bitcast ([2 x [3 x double]]* @_QMvarsEu to i8*), i64 24), i64 24, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %bytes, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @_QMmEa, i64 0, i64 0), i64 16, i1 false)
  ret void
}

; CHECK: callee - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{}
; CHECK-NEXT: {{(i8\*|ptr)}} %row: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT: {{(i8\*|ptr)}} %bytes: {[-1]:Pointer}
