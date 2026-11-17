; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; The call's function type differs from the callee's: an addrspace(10) pointer
; is passed to a parameter in addrspace 0. Julia 1.14 emits such direct calls to
; `jlcapi_` cfunction wrappers. The augmented and gradient calls must pass the
; addrspace(10) primal and shadow unchanged, like the original call (a bitcast
; between address spaces is invalid).

define double @sq(ptr %x) noinline {
entry:
  %v = load double, ptr %x, align 8
  %m = fmul double %v, %v
  store double %m, ptr %x, align 8
  ret double %m
}

define double @f(ptr addrspace(10) %x) {
entry:
  %a = call double @sq(ptr addrspace(10) %x)
  %b = call double @sq(ptr addrspace(10) %x)
  %r = fadd double %a, %b
  ret double %r
}

declare void @__enzyme_autodiff(...)

define void @test(ptr addrspace(10) %x, ptr addrspace(10) %dx) {
entry:
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr addrspace(10) %x, ptr addrspace(10) %dx)
  ret void
}

; CHECK: define internal void @diffef(ptr addrspace(10) {{.*}}%x, ptr addrspace(10) {{.*}}%"x'", double %differeturn)
; CHECK: %a_augmented = call {{.*}}double @augmented_sq(ptr addrspace(10) %x, ptr addrspace(10) %"x'")
; CHECK: call void @diffesq(ptr addrspace(10) %x, ptr addrspace(10) %"x'", double
; CHECK: call void @diffesq.{{[0-9]+}}(ptr addrspace(10) %x, ptr addrspace(10) %"x'", double
; CHECK-NOT: addrspacecast
