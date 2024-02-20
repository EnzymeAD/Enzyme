; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=f.preprocess -S | FileCheck %s; fi

declare void @free(ptr)

declare ptr @malloc(i64)

; This function just returns 2*input, its derivate should be 2.0.
define void @f.preprocess(ptr %param, i64 %mallocsize, ptr %res) {

  ; arithmetic block, changing anything here makes the bug go away
  %buffer1 = call ptr @malloc(i64 %mallocsize)
  %tmp = call ptr @malloc(i64 72)
  %ptrtoint = ptrtoint ptr %tmp to i64
  %and = and i64 %ptrtoint, -64
  %inttoptr = inttoptr i64 %and to ptr
  %loadarg = load double, ptr %param
  %storedargmul = fmul double %loadarg, 4.000000e+00
  store double %storedargmul, ptr %inttoptr
  call void @free(ptr %tmp)
  store double %storedargmul, ptr %buffer1

  ; prep arg 0 by setting the aligned pointer to the input
  %arg0 = alloca { ptr, ptr, i64 }
  %arg0_aligned = getelementptr inbounds { ptr, ptr, i64 }, ptr %arg0, i64 0, i32 1
  store ptr %param, ptr %arg0_aligned

  ; prep arg 1 by setting the aligned pointer to buffer1
  %arg1 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }
  %arg1_aligned = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %arg1, i64 0, i32 1
  store ptr %buffer1, ptr %arg1_aligned

  ; prep arg 2 by setting the aligned pointer to buffer2
  %arg2 = alloca { ptr, ptr, i64 }
  %arg2_aligned = getelementptr inbounds { ptr, ptr, i64 }, ptr %arg2, i64 0, i32 1
  %buffer2 = call ptr @malloc(i64 8)
  store ptr %buffer2, ptr %arg2_aligned

  ; nested call, required for bug
  call void @nested(ptr %arg0, ptr %arg1, ptr %arg2)

  ; return a result from this function, needs to be positioned after arithmetic block for bug
  %x = load double, ptr %param
  %y = fmul double %x, 2.0
  store double %y, ptr %res

  ret void
}

; Identity function, 2nd argument required for bug (but not used)
define void @nested(ptr %arg0, ptr %arg1, ptr %arg2) {

  ; load aligned pointer from %arg0 & load argument value
  %loadarg = load { ptr, ptr, i64 }, ptr %arg0
  %extractarg = extractvalue { ptr, ptr, i64 } %loadarg, 1
  %loadextractarg = load double, ptr %extractarg

  ; load aligned pointer from %arg2 & store result value
  %loadarg2 = load { ptr, ptr, i64 }, ptr %arg2
  %extractarg2 = extractvalue { ptr, ptr, i64 } %loadarg2, 1
  store double %loadextractarg, ptr %extractarg2

  ret void
}

; CHECK: ptr %param: icv:0
; CHECK-NEXT: i64 %mallocsize: icv:1
; CHECK-NEXT: ptr %res: icv:0

; CHECK: %buffer1 = call ptr @malloc(i64 %mallocsize): icv:0 ici:1
; CHECK-NEXT: %tmp = call ptr @malloc(i64 72): icv:1 ici:1
; CHECK-NEXT: %ptrtoint = ptrtoint ptr %tmp to i64: icv:1 ici:1
; CHECK-NEXT: %and = and i64 %ptrtoint, -64: icv:1 ici:1
; CHECK-NEXT: %inttoptr = inttoptr i64 %and to ptr: icv:1 ici:1
; CHECK-NEXT: %loadarg = load double, ptr %param, align 8: icv:0 ici:0
; CHECK-NEXT: %storedargmul = fmul double %loadarg, 4.000000e+00: icv:0 ici:0
; CHECK-NEXT: store double %storedargmul, ptr %inttoptr, align 8: icv:1 ici:1
; CHECK-NEXT: call void @free(ptr %tmp): icv:1 ici:1
; CHECK-NEXT: store double %storedargmul, ptr %buffer1, align 8: icv:1 ici:0
; CHECK-NEXT: %arg0 = alloca { ptr, ptr, i64 }, align 8: icv:0 ici:1
; CHECK-NEXT: %arg0_aligned = getelementptr inbounds { ptr, ptr, i64 }, ptr %arg0, i64 0, i32 1: icv:0 ici:1
; CHECK-NEXT: store ptr %param, ptr %arg0_aligned, align 8: icv:1 ici:0
; CHECK-NEXT: %arg1 = alloca { ptr, ptr, i64, [1 x i64], [1 x i64] }, align 8: icv:0 ici:1
; CHECK-NEXT: %arg1_aligned = getelementptr inbounds { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %arg1, i64 0, i32 1: icv:0 ici:1
; CHECK-NEXT: store ptr %buffer1, ptr %arg1_aligned, align 8: icv:1 ici:0
; CHECK-NEXT: %arg2 = alloca { ptr, ptr, i64 }, align 8: icv:0 ici:1
; CHECK-NEXT: %arg2_aligned = getelementptr inbounds { ptr, ptr, i64 }, ptr %arg2, i64 0, i32 1: icv:0 ici:1
; CHECK-NEXT: %buffer2 = call ptr @malloc(i64 8): icv:0 ici:1
; CHECK-NEXT: store ptr %buffer2, ptr %arg2_aligned, align 8: icv:1 ici:0
; CHECK-NEXT: call void @nested(ptr %arg0, ptr %arg1, ptr %arg2): icv:1 ici:0
; CHECK-NEXT: %x = load double, ptr %param, align 8: icv:0 ici:0
; CHECK-NEXT: %y = fmul double %x, 2.000000e+00: icv:0 ici:0
; CHECK-NEXT: store double %y, ptr %res, align 8: icv:1 ici:0
; CHECK-NEXT: ret void: icv:1 ici:1
