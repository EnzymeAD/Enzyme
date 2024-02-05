; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | llc -filetype=obj -o %t.o
; RUN: %clang %t.o -o %t && %run ./ %t | FileCheck %s

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@enzyme_dupnoneed = linkonce constant i8 0
@enzyme_const = linkonce constant i8 0

@.str = private unnamed_addr constant [3 x i8] c"%f\00"

declare i32 @printf(ptr, ...)

declare void @__enzyme_autodiff0(...)

declare void @free(ptr)

declare ptr @malloc(i64)

define i32 @main() {

  ; prep arguments & shadows
  %input = call ptr @malloc(i64 64)
  store double 0.2, ptr %input
  %in_shadow = call ptr @malloc(i64 8)
  store i64 0, ptr %in_shadow

  %output = call ptr @malloc(i64 8)
  %out_shadow = call ptr @malloc(i64 8)
  store double 1.000000e+00, ptr %out_shadow

  ; autodiff call
  call void (...) @__enzyme_autodiff0(ptr @f.preprocess, ptr %input, ptr %in_shadow, i64 1, ptr %output, ptr %out_shadow)

  ; print result
  %grad = load double, ptr %in_shadow
  call i32 (ptr, ...) @printf(ptr @.str, double %grad)

  call void @free(ptr %input)
  call void @free(ptr %in_shadow)
  call void @free(ptr %output)
  call void @free(ptr %out_shadow)

  ret i32 0
}

; This function just returns 2*input, its derivate should be 2.0.
define void @f.preprocess(ptr %0, i64 %1, ptr %2) {

  ; arithmetic block, changing anything here makes the bug go away
  %buffer1 = call ptr @malloc(i64 %1)
  %tmp = call ptr @malloc(i64 72)
  %4 = ptrtoint ptr %tmp to i64
  %5 = and i64 %4, -64
  %6 = inttoptr i64 %5 to ptr
  %7 = load double, ptr %0
  %8 = fmul double %7, 4.000000e+00
  store double %8, ptr %6
  call void @free(ptr %tmp)
  store double %8, ptr %buffer1

  ; prep arg 0 by setting the aligned pointer to the input
  %arg0 = alloca { ptr, ptr, i64 }
  %arg0_aligned = getelementptr inbounds { ptr, ptr, i64 }, ptr %arg0, i64 0, i32 1
  store ptr %0, ptr %arg0_aligned

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
  %x = load double, ptr %0
  %y = fmul double %x, 2.0
  store double %y, ptr %2

  ret void
}

; Identity function, 2nd argument required for bug
define void @nested(ptr %0, ptr %1, ptr %2) {

  ; load aligned pointer from %0 & load argument value
  %4 = load { ptr, ptr, i64 }, ptr %0
  %5 = extractvalue { ptr, ptr, i64 } %4, 1
  %6 = load double, ptr %5

  ; load aligned pointer from %2 & store result value
  %7 = load { ptr, ptr, i64 }, ptr %2
  %8 = extractvalue { ptr, ptr, i64 } %7, 1
  store double %6, ptr %8

  ret void
}