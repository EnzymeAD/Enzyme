; RUN: %opt %newLoadEnzyme -S -passes=enzyme-fixup-julia < %s 2>&1 | FileCheck %s

; The callee never returns: it always throws, as an augmented function Enzyme
; replaced by an error does. Its enzyme_sret holds a tracked pointer next to an
; untracked field, which on Julia 1.10 and 1.11 makes the GC lowering look for
; the roots in the second argument of an sret call. Nothing ever writes the roots
; of a function that does not return, so interprocedural passes may remove that
; argument; the buffer must therefore not be passed as sret, neither to the new
; function nor at its call sites.

; CHECK-NOT: warning:

; CHECK-LABEL: define void @caller(
; CHECK: %stack_sret = alloca { {{.*}} addrspace(10)*, i64 }
; CHECK: %stack_roots_AT = alloca [1 x {{.*}} addrspace(10)*]
; CHECK: call void @augmented_throws({ {{.*}} addrspace(10)*, i64 }* %stack_sret, [1 x {{.*}} addrspace(10)*]* "enzymejl_returnRoots"="1" %stack_roots_AT, {} addrspace(10)* %v)
; CHECK-NOT: sret(

; CHECK-LABEL: define internal void @augmented_throws({ {{.*}} addrspace(10)*, i64 }* noalias %0, [1 x {{.*}} addrspace(10)*]* noalias writeonly "enzymejl_returnRoots"="1" %1, {} addrspace(10)* %v)
; CHECK-NOT: ret
; CHECK: unreachable

declare void @jl_throw({} addrspace(12)*) #0

define void @caller({} addrspace(10)* %v) {
entry:
  %sret = alloca { {} addrspace(10)*, i64 }, align 8
  call void @augmented_throws({ {} addrspace(10)*, i64 }* "enzyme_sret"="test_type6" %sret, {} addrspace(10)* %v)
  unreachable
}

define internal void @augmented_throws({ {} addrspace(10)*, i64 }* "enzyme_sret"="test_type6" %sret, {} addrspace(10)* %v) {
top:
  %p = getelementptr inbounds { {} addrspace(10)*, i64 }, { {} addrspace(10)*, i64 }* %sret, i64 0, i32 0
  store {} addrspace(10)* %v, {} addrspace(10)** %p, align 8
  %e = addrspacecast {} addrspace(10)* %v to {} addrspace(12)*
  call void @jl_throw({} addrspace(12)* %e)
  unreachable
}

attributes #0 = { noreturn }
