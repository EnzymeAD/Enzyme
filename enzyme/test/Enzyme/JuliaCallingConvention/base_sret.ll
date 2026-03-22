; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_sret(ptr noalias sret({ {{.*}} }) %0, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[LOAD:%[a-zA-Z_0-9]+]] = load {{.*}}, ptr %arg, align 8
; CHECK-NEXT:   store {{.*}} [[LOAD]], ptr %0, align 8
; CHECK-NEXT:   ret void

%jl_value = type opaque

define void @test_sret({ %jl_value addrspace(10)* }* sret({ %jl_value addrspace(10)* }) %sret, %jl_value addrspace(10)** %arg) {
entry:
  %val = load %jl_value addrspace(10)*, %jl_value addrspace(10)** %arg, align 8
  store %jl_value addrspace(10)* %val, { %jl_value addrspace(10)* }* %sret, align 8
  ret void
}
