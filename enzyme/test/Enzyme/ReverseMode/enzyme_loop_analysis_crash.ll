; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

define hidden void @_Z10entry_wrapRN6enzyme6tensorIfJLm2ELm3EEEES2_RKS1_(ptr %out_0, ptr %out_1, ptr %in_0) #0 {
entry:
  %out_0.addr = alloca ptr, align 8
  %out_1.addr = alloca ptr, align 8
  %in_0.addr = alloca ptr, align 8
  store ptr %out_0, ptr %out_0.addr, align 8
  store ptr %out_1, ptr %out_1.addr, align 8
  store ptr %in_0, ptr %in_0.addr, align 8
  %0 = load ptr, ptr %out_0.addr, align 8
  %1 = load ptr, ptr %out_1.addr, align 8
  %2 = load ptr, ptr %in_0.addr, align 8
  call void @_Z4myfnILm2ELm3EEvRN6enzyme6tensorIfJXT_EXT0_EEEES3_RKS2_(ptr %0, ptr %1, ptr %2)
  ret void
}

define hidden void @_Z4myfnILm2ELm3EEvRN6enzyme6tensorIfJXT_EXT0_EEEES3_RKS2_(ptr %0, ptr %1, ptr %2) #0 {
entry:
  ret void
}

declare void @__enzyme_autodiff(...)

define void @test_derivative(ptr %out_0, ptr %out_0_d, ptr %out_1, ptr %out_1_d, ptr %in_0, ptr %in_0_d) {
entry:
  call void (...) @__enzyme_autodiff(ptr @_Z10entry_wrapRN6enzyme6tensorIfJLm2ELm3EEEES2_RKS1_, ptr %out_0, ptr %out_0_d, ptr %out_1, ptr %out_1_d, ptr %in_0, ptr %in_0_d)
  ret void
}

attributes #0 = { mustprogress nounwind }

; CHECK: define internal void @diffe_Z10entry_wrap
