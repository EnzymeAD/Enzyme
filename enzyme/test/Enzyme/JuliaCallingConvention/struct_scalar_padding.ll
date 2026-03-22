; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_scalar_padding(ptr noalias sret(%outer_struct) %0, ptr noalias writeonly "enzymejl_returnRoots"="2" %1, ptr %arg)

%outer_struct = type { i64, i8 addrspace(10)*, i32, i8 addrspace(10)* }

define void @test_scalar_padding(%outer_struct* sret(%outer_struct) %sret, %outer_struct* %arg) {
entry:
  %val = load %outer_struct, %outer_struct* %arg, align 8
  store %outer_struct %val, %outer_struct* %sret, align 8
  ret void
}
