; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_scalar_padding(ptr noalias sret({ i64, ptr addrspace(10), i32, ptr addrspace(10) }) %sret, ptr noalias writeonly "enzymejl_returnRoots"="2" %0, ptr %arg)

%outer_struct = type { i64, ptr addrspace(10), i32, ptr addrspace(10) }

define void @test_scalar_padding(ptr sret(%outer_struct) %sret, ptr %arg) {
entry:
  %val = load %outer_struct, ptr %arg, align 8
  store %outer_struct %val, ptr %sret, align 8
  ret void
}
