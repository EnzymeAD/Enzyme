; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s

; CHECK-LABEL: define void @test_rereturn(ptr noalias sret({ { ptr addrspace(10) }, [1 x ptr addrspace(10)] }) %0, ptr %arg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[GEP1:%[0-9]+]] = getelementptr inbounds { { ptr addrspace(10) }, [1 x ptr addrspace(10)] }, ptr %0, i32 0, i32 1
; CHECK-NEXT:   [[GEP2:%[0-9]+]] = getelementptr inbounds { { ptr addrspace(10) }, [1 x ptr addrspace(10)] }, ptr %0, i32 0, i32 0
; CHECK-NEXT:   store { ptr addrspace(10) } zeroinitializer, ptr [[GEP2]], align 8
; CHECK-NEXT:   ret void

define { ptr addrspace(10) } @test_rereturn(ptr %arg, ptr "enzymejl_returnRoots"="1" %roots) {
entry:
  ret { ptr addrspace(10) } zeroinitializer
}
