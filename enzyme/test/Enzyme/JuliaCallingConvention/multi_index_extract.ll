; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_multi_index(ptr noalias sret({{.*}}) %0, ptr noalias writeonly "enzymejl_returnRoots"="2" %1, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[LOAD:%[a-zA-Z_0-9]+]] = load {{.*}}, ptr %arg, align 8
; CHECK-NEXT:   store {{.*}} [[LOAD]], ptr %0, align 8
; CHECK-NEXT:   [[EXT1:%[a-zA-Z_0-9]+]] = extractvalue {{.*}} [[LOAD]], 0, 0
; CHECK-NEXT:   [[EXT2:%[a-zA-Z_0-9]+]] = extractvalue {{.*}} [[LOAD]], 0, 1
; CHECK-NEXT:   ret void

%inner_struct = type { ptr addrspace(10), ptr addrspace(10) }
%outer_struct = type { %inner_struct, i8 }

define void @test_multi_index(ptr sret(%outer_struct) %sret, ptr %arg) {
entry:
  %val = load %outer_struct, ptr %arg, align 8
  store %outer_struct %val, ptr %sret, align 8

  ; Extract both pointers from inner struct to achieve full path coverage.
  %p1 = extractvalue %outer_struct %val, 0, 0
  %p2 = extractvalue %outer_struct %val, 0, 1

  ret void
}
