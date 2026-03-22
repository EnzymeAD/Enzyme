; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_sret(ptr noalias sret({ {{.*}} }) %0, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[LOAD:%[a-zA-Z_0-9]+]] = load {{.*}}, ptr %arg, align 8
; CHECK-NEXT:   [[GEP:%[a-zA-Z_0-9]+]] = getelementptr inbounds {{.*}}, ptr %0, i32 0, i32 0
; CHECK-NEXT:   store {{.*}} [[LOAD]], ptr [[GEP]], align 8
; CHECK-NEXT:   ret void

define void @test_sret({ {} addrspace(10)* }* sret({ {} addrspace(10)* }) %sret, {} addrspace(10)** %arg) {
entry:
  %val = load {} addrspace(10)*, {} addrspace(10)** %arg, align 8
  %gep = getelementptr inbounds { {} addrspace(10)* }, { {} addrspace(10)* }* %sret, i32 0, i32 0
  store {} addrspace(10)* %val, {} addrspace(10)** %gep, align 8
  ret void
}
