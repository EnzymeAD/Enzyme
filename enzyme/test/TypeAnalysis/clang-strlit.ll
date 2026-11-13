; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -S -o /dev/null | FileCheck %s

@.str = private unnamed_addr constant [8 x i8] c"alpha_x\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"alpha_y\00", align 1
@"??_C@_07ABCDEFGH@alpha_z?$AA@" = linkonce_odr unnamed_addr constant [8 x i8] c"alpha_z\00", align 1
@__const.callee.raw = private unnamed_addr constant [8 x i8] c"\00\00\00\00\00\00\F0?", align 1

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

define void @callee(ptr %buf1, ptr %buf2, ptr %buf3, ptr %buf4) {
entry:
  %ld = load i64, ptr @.str, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %buf1, ptr @.str, i64 3, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %buf2, ptr @.str.1, i64 3, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %buf3, ptr @"??_C@_07ABCDEFGH@alpha_z?$AA@", i64 3, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %buf4, ptr @__const.callee.raw, i64 3, i1 false)
  ret void
}

; CHECK: callee - {} |{[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Pointer}:{} {[-1]:Pointer}:{}
; CHECK-NEXT: ptr %buf1: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT: ptr %buf2: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT: ptr %buf3: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer}
; CHECK-NEXT: ptr %buf4: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   %ld = load i64, ptr @.str, align 1: {[-1]:Integer}
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %buf1, ptr @.str, i64 3, i1 false): {}
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %buf2, ptr @.str.1, i64 3, i1 false): {}
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %buf3, ptr @"??_C@_07ABCDEFGH@alpha_z?$AA@", i64 3, i1 false): {}
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %buf4, ptr @__const.callee.raw, i64 3, i1 false): {}
; CHECK-NEXT:   ret void: {}
