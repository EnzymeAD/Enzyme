; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-runtime-error -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-runtime-error -passes="enzyme" -S | FileCheck %s

; With one end on the loop back edge and one on the exit, a single end at the
; exit would close only the last iteration's region.  The ends must stay apart,
; and reversing the region is then an error rather than a silently wrong
; preserve.

declare token @llvm.julia.gc_preserve_begin(...)

declare void @llvm.julia.gc_preserve_end(token)

define void @loop({} addrspace(10)* addrspace(10)* %arr, i64 %n) {
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %inc, %a ]
  %p = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(10)* %arr, i64 %i
  %z = load {} addrspace(10)*, {} addrspace(10)* addrspace(10)* %p, align 8
  %tok = call token (...) @llvm.julia.gc_preserve_begin({} addrspace(10)* %z)
  %z_flt = bitcast {} addrspace(10)* %z to double addrspace(10)*
  %v = load double, double addrspace(10)* %z_flt, align 8
  %m = fmul double %v, %v
  store double %m, double addrspace(10)* %z_flt, align 8
  %inc = add i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %b, label %a

a:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %header

b:
  call void @llvm.julia.gc_preserve_end(token %tok)
  br label %exit

exit:
  ret void
}

; Function Attrs: nounwind
declare i8* @__enzyme_virtualreverse(...)

define i8* @test() {
entry:
  %0 = call i8* (...) @__enzyme_virtualreverse(void ({} addrspace(10)* addrspace(10)*, i64)* @loop)
  ret i8* %0
}

; CHECK-LABEL: define internal {{(i8\*|ptr)}} @augmented_loop(
; CHECK: %[[TOK:.+]] = call token (...) @llvm.julia.gc_preserve_begin(
; CHECK: a:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[TOK]])
; CHECK-NEXT: br label %header
; CHECK: b:
; CHECK-NEXT: call void @llvm.julia.gc_preserve_end(token %[[TOK]])

; CHECK-LABEL: define internal void @diffeloop(
; CHECK-NOT: @llvm.julia.gc_preserve_begin()
; CHECK: invertentry:
; CHECK-NOT: @llvm.julia.gc_preserve_end
; CHECK: inverta:
; CHECK: call i32 @puts(
; CHECK-NEXT: call void @exit(i32 1)
; CHECK: invertb:
; CHECK: call i32 @puts(
; CHECK-NEXT: call void @exit(i32 1)
