; RUN: if [ %llvmver -lt 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare i32 @cudaMalloc(double**, i64)
declare i32 @posix_memalign(double**, i64, i64)
declare i32 @cuMemAlloc_v2(i64*, i64)
declare i32 @posix_memalign_as1(i8 addrspace(1)**, i64, i64) "enzyme_math"="posix_memalign"
declare double @__enzyme_autodiff(...)

define double @f_cuda_typed(double %x) {
entry:
  %pp = alloca double*, align 8
  %a = call i32 @cudaMalloc(double** %pp, i64 8)
  %p = load double*, double** %pp, align 8
  store double %x, double* %p, align 8
  %v = load double, double* %p, align 8
  ret double %v
}

define double @f_posix_typed(double %x) {
entry:
  %pp = alloca double*, align 8
  %a = call i32 @posix_memalign(double** %pp, i64 8, i64 8)
  %p = load double*, double** %pp, align 8
  store double %x, double* %p, align 8
  %v = load double, double* %p, align 8
  ret double %v
}

define double @f_driver_typed(double %x) {
entry:
  %pp = alloca i64, align 8
  %a = call i32 @cuMemAlloc_v2(i64* %pp, i64 8)
  %d = load i64, i64* %pp, align 8
  %p = inttoptr i64 %d to double*
  store double %x, double* %p, align 8
  %v = load double, double* %p, align 8
  ret double %v
}

define double @f_posix_as1(double %x) {
entry:
  %pp = alloca i8 addrspace(1)*, align 8
  %a = call i32 @posix_memalign_as1(i8 addrspace(1)** %pp, i64 8, i64 8)
  %p = load i8 addrspace(1)*, i8 addrspace(1)** %pp, align 8
  %pd = bitcast i8 addrspace(1)* %p to double addrspace(1)*
  store double %x, double addrspace(1)* %pd, align 8
  %v = load double, double addrspace(1)* %pd, align 8
  ret double %v
}

define double @test_cuda_typed(double %x) {
  %r = call double (...) @__enzyme_autodiff(double (double)* @f_cuda_typed, double %x)
  ret double %r
}

define double @test_posix_typed(double %x) {
  %r = call double (...) @__enzyme_autodiff(double (double)* @f_posix_typed, double %x)
  ret double %r
}

define double @test_driver_typed(double %x) {
  %r = call double (...) @__enzyme_autodiff(double (double)* @f_driver_typed, double %x)
  ret double %r
}

define double @test_posix_as1(double %x) {
  %r = call double (...) @__enzyme_autodiff(double (double)* @f_posix_as1, double %x)
  ret double %r
}

; CHECK: define internal { double } @diffef_cuda_typed(
; CHECK:   call i32 @cudaMalloc(double** %"pp'ipa", i64 8)
; CHECK:   %[[C0:.+]] = bitcast double* %{{.+}} to i8*
; CHECK:   call i32 @cudaMemset(i8* nonnull %[[C0]], i32 0, i64 8)
; CHECK: invertentry:
; CHECK:   call i32 @cudaFree(i8* %{{.+}})
; CHECK:   call i32 @cudaFree(i8* %{{.+}})

; CHECK: define internal { double } @diffef_posix_typed(
; CHECK:   call i32 @posix_memalign(double** %"pp'ipa", i64 8, i64 8)
; CHECK:   call void @llvm.memset.p0i8.i64(i8* %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: invertentry:
; CHECK:   call void @free(i8* %{{.+}})
; CHECK:   call void @free(i8* %{{.+}})

; CHECK: define internal { double } @diffef_driver_typed(
; CHECK:   call i32 @cuMemAlloc_v2(i64* %"pp'ipa", i64 8)
; CHECK:   call i32 @cuMemsetD8_v2(i64 %{{.+}}, i8 0, i64 8)
; CHECK: invertentry:
; CHECK:   call i32 @cuMemFree_v2(i64 %{{.+}})
; CHECK:   call i32 @cuMemFree_v2(i64 %{{.+}})

; CHECK: define internal { double } @diffef_posix_as1(
; CHECK:   call void @llvm.memset.p1i8.i64(i8 addrspace(1)* %{{.+}}, i8 0, i64 8, i1 false)
; CHECK: invertentry:
; CHECK:   %[[AS0:.+]] = addrspacecast i8 addrspace(1)* %{{.+}} to i8*
; CHECK:   call void @free(i8* %[[AS0]])
; CHECK:   %[[AS1:.+]] = addrspacecast i8 addrspace(1)* %{{.+}} to i8*
; CHECK:   call void @free(i8* %[[AS1]])
