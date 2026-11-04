; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

@_QQclX3132333435363738 = linkonce_odr constant [8 x i8] c"12345678", align 1

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

define void @callee(i8* %buf) {
entry:
  %ld = load i64, i64* bitcast ([8 x i8]* @_QQclX3132333435363738 to i64*), align 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_QQclX3132333435363738, i64 0, i64 0), i64 3, i1 false)
  ret void
}

; CHECK: callee - {} |{[-1]:Pointer}:{}
; CHECK-NEXT: i8* %buf: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ld = load i64, i64* bitcast ([8 x i8]* @_QQclX3132333435363738 to i64*), align 1: {[-1]:Integer}
; CHECK-NEXT:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_QQclX3132333435363738, i64 0, i64 0), i64 3, i1 false): {}
; CHECK-NEXT:   ret void: {}
