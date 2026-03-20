; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

define hidden void @_Z10entry_wrapRN6enzyme6tensorIfJLm2ELm3EEEES2_RKS1_(i8* %out_0, i8* %out_1, i8* %in_0) #0 {
entry:
  %out_0.addr = alloca i8*, align 8
  %out_1.addr = alloca i8*, align 8
  %in_0.addr = alloca i8*, align 8
  store i8* %out_0, i8** %out_0.addr, align 8
  store i8* %out_1, i8** %out_1.addr, align 8
  store i8* %in_0, i8** %in_0.addr, align 8
  %0 = load i8*, i8** %out_0.addr, align 8
  %1 = load i8*, i8** %out_1.addr, align 8
  %2 = load i8*, i8** %in_0.addr, align 8
  call void @_Z4myfnILm2ELm3EEvRN6enzyme6tensorIfJXT_EXT0_EEEES3_RKS2_(i8* %0, i8* %1, i8* %2)
  ret void
}

define hidden void @_Z4myfnILm2ELm3EEvRN6enzyme6tensorIfJXT_EXT0_EEEES3_RKS2_(i8* %0, i8* %1, i8* %2) #0 {
entry:
  ret void
}

declare void @__enzyme_autodiff(...)

define void @test_derivative(i8* %out_0, i8* %out_0_d, i8* %out_1, i8* %out_1_d, i8* %in_0, i8* %in_0_d) {
entry:
  call void (...) @__enzyme_autodiff(i8* bitcast (void (i8*, i8*, i8*)* @_Z10entry_wrapRN6enzyme6tensorIfJLm2ELm3EEEES2_RKS1_ to i8*), i8* %out_0, i8* %out_0_d, i8* %out_1, i8* %out_1_d, i8* %in_0, i8* %in_0_d)
  ret void
}

attributes #0 = { mustprogress nounwind }

; CHECK: define internal void @diffe_Z10entry_wrap
