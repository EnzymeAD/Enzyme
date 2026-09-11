; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

; A gc_preserve_begin may legally end up with several gc_preserve_ends: the
; token it produces may have many uses, so LLVM is free to clone a block holding
; one of the ends. Reverse mode mirrors the region by emitting a begin at the
; invert of the end and the end at the invert of the begin, which only
; type-checks if a single end post-dominates the begin, so the ends have to be
; merged before differentiating. Both functions below abort in the verifier if
; they are not.

declare token @llvm.julia.gc_preserve_begin(...)

declare void @llvm.julia.gc_preserve_end(token)

; The two ends rejoin at a block the begin dominates.
define void @f({} addrspace(10)* %z, i1 %c) {
entry:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

join:
  ret void
}

; The two ends only rejoin at a block that is also reachable without ever
; running the begin, so the edges coming out of the region have to be split off
; into a block of their own first.
define void @g({} addrspace(10)* %z, i1 %c, i1 %d) {
entry:
  br i1 %d, label %pre, label %join

pre:
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  store double 3.140000e+00, double addrspace(10)* %z_flt, align 8
  br i1 %c, label %a, label %b

a:
  store double 2.710000e+00, double addrspace(10)* %z_flt, align 8
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %join

join:
  ret void
}

; Function Attrs: nounwind
declare i8* @__enzyme_virtualreverse(...)

define i8* @test({} addrspace(10)* %z, i1 %c, i1 %d) {
entry:
  %0 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1)* @f)
  %1 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)*, i1, i1)* @g)
  ret i8* %0
}

; CHECK: define internal {{(i8\*|ptr)}} @augmented_f(
; CHECK: define internal void @diffef(
; CHECK: define internal {{(i8\*|ptr)}} @augmented_g(
; CHECK: define internal void @diffeg(
