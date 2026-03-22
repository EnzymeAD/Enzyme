; RUN: %opt %loadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_vector(ptr noalias sret(<2 x ptr addrspace(10)>) %sret, ptr noalias writeonly "enzymejl_returnRoots"="2" %0, ptr %arg)

define void @test_vector(ptr sret(<2 x ptr addrspace(10)>) %sret, ptr %arg) {
entry:
  %val = load <2 x ptr addrspace(10)>, ptr %arg, align 16
  store <2 x ptr addrspace(10)> %val, ptr %sret, align 16
  ret void
}
