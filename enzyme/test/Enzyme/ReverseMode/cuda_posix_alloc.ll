; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare i32 @posix_memalign(ptr, i64, i64)
declare i32 @cudaMalloc(ptr, i64)
declare i32 @cudaMemset(ptr, i32, i64)
declare i32 @cudaFree(ptr)
declare i32 @cuMemAlloc_v2(ptr, i64)
declare i32 @cuMemsetD8_v2(i64, i8, i64)
declare i32 @cuMemFree_v2(i64)

declare [2 x double] @__enzyme_autodiff_vec(...)
declare double @__enzyme_autodiff(...)
declare { ptr, double } @__enzyme_augmentfwd(...)
declare [2 x double] @__enzyme_reverse(...)

define double @posix_loop(double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %sum = phi double [ 0.0, %entry ], [ %nsum, %loop ]
  %pp = alloca ptr, align 8
  %a = call i32 @posix_memalign(ptr %pp, i64 8, i64 8)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %nsum = fadd double %sum, %v
  %inc = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %inc, %n
  br i1 %cmp, label %exit, label %loop

exit:
  ret double %nsum
}

define [2 x double] @test_posix_vecrev_loop(double %x, i64 %n) {
  %r = call [2 x double] (...) @__enzyme_autodiff_vec(ptr @posix_loop, metadata !"enzyme_width", i64 2, double %x, metadata !"enzyme_const", i64 %n)
  ret [2 x double] %r
}

define double @driver_alloc(double %x) {
entry:
  %pp = alloca i64, align 8
  %a = call i32 @cuMemAlloc_v2(ptr %pp, i64 8)
  %d = load i64, ptr %pp, align 8
  %p = inttoptr i64 %d to ptr
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %f = call i32 @cuMemFree_v2(i64 %d)
  ret double %v
}

define double @test_driver_alloc(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @driver_alloc, double %x)
  ret double %r
}

define double @runtime_alloc(double %x) {
entry:
  %pp = alloca ptr, align 8
  %a = call i32 @cudaMalloc(ptr %pp, i64 8)
  %p = load ptr, ptr %pp, align 8
  store double %x, ptr %p, align 8
  %v = load double, ptr %p, align 8
  %f = call i32 @cudaFree(ptr %p)
  ret double %v
}

define double @test_runtime_alloc(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @runtime_alloc, double %x)
  ret double %r
}

define [2 x double] @test_runtime_split_vec(double %x) {
  %aug = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @runtime_alloc, metadata !"enzyme_width", i64 2, double %x)
  %tape = extractvalue { ptr, double } %aug, 0
  %r = call [2 x double] (...) @__enzyme_reverse(ptr @runtime_alloc, metadata !"enzyme_width", i64 2, double %x, [2 x double] [double 1.0, double 2.0], ptr %tape)
  ret [2 x double] %r
}

; CHECK: define internal { [2 x double] } @diffe2posix_loop(
; CHECK: invertloop:
; CHECK:   %[[CACHEPTR:.+]] = load ptr, ptr %_cache, align 8
; CHECK-NEXT:   %[[CACHEGEP:.+]] = getelementptr inbounds [2 x ptr], ptr %[[CACHEPTR]], i64 %{{.+}}
; CHECK-NEXT:   %[[TOFREE:.+]] = load [2 x ptr], ptr %[[CACHEGEP]], align 8
; CHECK-NEXT:   %[[F0:.+]] = extractvalue [2 x ptr] %[[TOFREE]], 0
; CHECK-NEXT:   call void @free(ptr %[[F0]])
; CHECK-NEXT:   %[[F1:.+]] = extractvalue [2 x ptr] %[[TOFREE]], 1
; CHECK-NEXT:   call void @free(ptr %[[F1]])

; CHECK: define internal { double } @diffedriver_alloc(
; CHECK:   call i32 @cuMemAlloc_v2(ptr %"pp'ipa", i64 8)
; CHECK:   %[[DPTR:.+]] = ptrtoint ptr %{{.+}} to i64
; CHECK:   call i32 @cuMemsetD8_v2(i64 %[[DPTR]], i8 0, i64 8)
; CHECK: invertentry:
; CHECK:   call i32 @cuMemFree_v2(i64 %{{.+}})
; CHECK:   call i32 @cuMemFree_v2(i64 %{{.+}})
; CHECK-NOT: @cuMemFree(

; CHECK: define internal { double } @differuntime_alloc(
; CHECK:   call i32 @cudaMalloc(ptr %"pp'ipa", i64 8)
; CHECK:   call i32 @cudaMemset(ptr nonnull %{{.+}}, i32 0, i64 8)
; CHECK: invertentry:
; CHECK:   call i32 @cudaFree(ptr %{{.+}})
; CHECK:   call i32 @cudaFree(ptr %{{.+}})

; CHECK: define internal { [2 x double] } @diffe2runtime_alloc(
; CHECK: invertentry:
; CHECK:   %[[SF0:.+]] = extractvalue { [2 x ptr], [2 x ptr], ptr } %truetape, 0, 0
; CHECK:   call i32 @cudaFree(ptr %[[SF0]])
; CHECK:   %[[SF1:.+]] = extractvalue { [2 x ptr], [2 x ptr], ptr } %truetape, 0, 1
; CHECK:   call i32 @cudaFree(ptr %[[SF1]])
