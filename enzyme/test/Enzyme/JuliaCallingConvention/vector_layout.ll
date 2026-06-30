; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_vector({{.*}} noalias sret(<2 x {{.*}}>) %0, {{.*}} %arg)

define void @test_vector(<2 x i8 addrspace(10)*>* sret(<2 x i8 addrspace(10)*>) %sret, <2 x i8 addrspace(10)*>* %arg) {
entry:
  %val = load <2 x i8 addrspace(10)*>, <2 x i8 addrspace(10)*>* %arg, align 16
  store <2 x i8 addrspace(10)*> %val, <2 x i8 addrspace(10)*>* %sret, align 16
  ret void
}
