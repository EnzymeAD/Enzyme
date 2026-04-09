; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_scalar_padding({{.*}} noalias sret(%outer_struct) %0, {{.*}} noalias writeonly "enzymejl_returnRoots"="2" %1, {{.*}} %arg)

%outer_struct = type { i64, {} addrspace(10)*, i32, {} addrspace(10)* }

define void @test_scalar_padding(%outer_struct* sret(%outer_struct) %sret, %outer_struct* %arg) {
entry:
  %val = load %outer_struct, %outer_struct* %arg, align 8
  store %outer_struct %val, %outer_struct* %sret, align 8
  ret void
}
