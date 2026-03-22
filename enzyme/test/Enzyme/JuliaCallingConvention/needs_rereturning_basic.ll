; RUN: %opt %loadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_rereturn(ptr sret({ { ptr addrspace(10) }, [1 x ptr addrspace(10)] }) %0, ptr noalias writeonly "enzymejl_returnRoots"="1" %1, ptr %arg, ptr "enzymejl_returnRoots"="1" %roots)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[GEP:%[0-9]+]] = getelementptr { ptr addrspace(10) }, ptr %0, i64 0
; CHECK-NEXT:   store { ptr addrspace(10) } zeroinitializer, ptr [[GEP]], align 8
; CHECK-NEXT:   ret void

define { ptr addrspace(10) } @test_rereturn(ptr %arg, ptr "enzymejl_returnRoots"="1" %roots) {
entry:
  ret { ptr addrspace(10) } zeroinitializer
}
