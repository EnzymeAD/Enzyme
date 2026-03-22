; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_multi_index(ptr noalias sret({{.*}}) %0, ptr noalias writeonly "enzymejl_returnRoots"="2" %1, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[LOAD:%[a-zA-Z_0-9]+]] = load {{.*}}, ptr %arg, align 8
; CHECK-NEXT:   store {{.*}} [[LOAD]], ptr %0, align 8
; CHECK-NEXT:   [[EXT1:%[a-zA-Z_0-9]+]] = extractvalue {{.*}} [[LOAD]], 0, 0
; CHECK-NEXT:   [[EXT2:%[a-zA-Z_0-9]+]] = extractvalue {{.*}} [[LOAD]], 0, 1
; CHECK-NEXT:   [[GEP1:%[a-zA-Z_0-9]+]] = getelementptr inbounds {{.*}}, ptr %1, i64 0, i32 0
; CHECK-NEXT:   [[GEP2:%[a-zA-Z_0-9]+]] = getelementptr inbounds {{.*}}, ptr %0, i64 0, i32 0, i32 0
; CHECK-NEXT:   [[LOAD1:%[a-zA-Z_0-9]+]] = load {{.*}}, ptr [[GEP2]], align 8
; CHECK-NEXT:   store {{.*}} [[LOAD1]], ptr [[GEP1]], align 8
; CHECK-NEXT:   [[GEP3:%[a-zA-Z_0-9]+]] = getelementptr inbounds {{.*}}, ptr %1, i64 0, i32 1
; CHECK-NEXT:   [[GEP4:%[a-zA-Z_0-9]+]] = getelementptr inbounds {{.*}}, ptr %0, i64 0, i32 0, i32 1
; CHECK-NEXT:   [[LOAD2:%[a-zA-Z_0-9]+]] = load {{.*}}, ptr [[GEP4]], align 8
; CHECK-NEXT:   store {{.*}} [[LOAD2]], ptr [[GEP3]], align 8
; CHECK-NEXT:   ret void

%inner_struct = type { i8 addrspace(10)*, i8 addrspace(10)* }
%outer_struct = type { %inner_struct, i8 }

define void @test_multi_index(%outer_struct* sret(%outer_struct) %sret, %outer_struct* %arg) {
entry:
  %val = load %outer_struct, %outer_struct* %arg, align 8
  store %outer_struct %val, %outer_struct* %sret, align 8

  ; Extract both pointers from inner struct to achieve full path coverage.
  %p1 = extractvalue %outer_struct %val, 0, 0
  %p2 = extractvalue %outer_struct %val, 0, 1

  ret void
}
