; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; A gc preserve region may be closed by more than one gc_preserve_end -- LICM
; hoisting the preserve of a loop invariant value out of a loop while
; duplicating the end onto every loop exit produces exactly that. Such a region
; cannot be reversed by opening it at the reverse of an end: every end is a
; separate entry of the reversed region, so the re-created gc_preserve_end would
; have to consume a phi of the tokens, and tokens cannot be phi'd. Check that a
; self contained preserve is emitted at the reverse of the begin instead of a
; token which does not dominate its use.

declare token @llvm.julia.gc_preserve_begin(...)

declare void @llvm.julia.gc_preserve_end(token)

define void @f(ptr addrspace(10) %z, i1 %c) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %z)
  br i1 %c, label %then, label %else

then:
  store double 3.140000e+00, ptr addrspace(10) %z, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %exit

else:
  store double 2.710000e+00, ptr addrspace(10) %z, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %exit

exit:
  ret void
}

; Function Attrs: nounwind
declare ptr @__enzyme_virtualreverse(...)

define ptr @test() {
entry:
  %0 = call ptr (...) @__enzyme_virtualreverse(ptr @f)
  ret ptr %0
}

; CHECK: define internal ptr @augmented_f(ptr addrspace(10) %z, ptr addrspace(10) %"z'", i1 %c)
; CHECK: entry:
; CHECK:   %[[augtok:.+]] = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %z, ptr addrspace(10) %"z'")
; CHECK:   br i1 %c, label %then, label %else

; CHECK: then:
; CHECK:   store double 3.140000e+00, ptr addrspace(10) %z, align 8
; CHECK:   call void @llvm.julia.gc_preserve_end(token %[[augtok]])

; CHECK: else:
; CHECK:   store double 2.710000e+00, ptr addrspace(10) %z, align 8
; CHECK:   call void @llvm.julia.gc_preserve_end(token %[[augtok]])

; CHECK: define internal void @diffef(ptr addrspace(10) %z, ptr addrspace(10) %"z'", i1 %c, ptr %tapeArg)

; The reverse of the begin is the single exit of the reversed region, so the
; preserve is opened and closed there rather than spanning it.
; CHECK: invertentry:
; CHECK-NEXT:   %[[revtok:.+]] = call token (...) @llvm.julia.gc_preserve_begin(ptr addrspace(10) %z, ptr addrspace(10) %"z'")
; CHECK-NEXT:   call void @llvm.julia.gc_preserve_end(token %[[revtok]])
; CHECK-NEXT:   ret void

; CHECK: invertthen:
; CHECK-NEXT:   store double 0.000000e+00, ptr addrspace(10) %"z'"

; CHECK: invertelse:
; CHECK-NEXT:   store double 0.000000e+00, ptr addrspace(10) %"z'"
