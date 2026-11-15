; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -mtriple=nvptx64-nvidia-cuda -enzyme-preopt=false -enzyme -mem2reg -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -mtriple=nvptx64-nvidia-cuda -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify),verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -mtriple=nvptx64-nvidia-cuda -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify),verify" -S | sed '/^target triple/d' | %lli -

; Regression test for https://github.com/EnzymeAD/Enzyme/issues/3219:
; on NVPTX targets, exponential cache growth allocates a replacement buffer and
; copies prior entries via memcpy instead of calling realloc, so it must free
; the superseded buffer on every growth after the initial allocation.

declare double @__enzyme_autodiff(ptr, ...)

; CHECK-LABEL: define internal { double } @diffef(double %x, double %differeturn)
; CHECK: loop:
; CHECK:   %[[CACHE:.+]] = phi ptr [ null, %entry ], [ %[[NEXT_CACHE:.+]], %__enzyme_exponentialallocation.exit ]
; CHECK: grow.i:
; CHECK:   %[[FIRST:.+]] = icmp eq i64 %{{.+}}, 1
; CHECK:   %[[NEWBUF:.+]] = call noalias nonnull ptr @malloc(i64 %{{.+}})
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr %[[NEWBUF]], ptr %[[CACHE]], i64 %{{.+}}, i1 false)
; CHECK-NEXT:   br i1 %[[FIRST]], label %afterfree.i, label %freeprev.i
; CHECK: freeprev.i:
; CHECK-NEXT:   call void @free(ptr nonnull %[[CACHE]])
; CHECK-NEXT:   br label %afterfree.i
; CHECK: afterfree.i:
; CHECK-NEXT:   br label %__enzyme_exponentialallocation.exit
; CHECK: __enzyme_exponentialallocation.exit:
; CHECK-NEXT:   %[[NEXT_CACHE]] = phi ptr [ %[[NEWBUF]], %afterfree.i ], [ %[[CACHE]], %loop ]
; CHECK: invertentry:
; CHECK:   call void @free(ptr nonnull %[[NEXT_CACHE]])
define double @f(double %x) {
entry:
  br label %loop

loop:
  %s = phi double [ 1.000000e+00, %entry ], [ %next, %loop ]
  %next = fmul double %s, %x
  %done = fcmp olt double %next, 1.000000e-01
  br i1 %done, label %exit, label %loop

exit:
  ret double %next
}

define double @df(double %x) {
entry:
  %dx = call double (ptr, ...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %dx
}

define i32 @main() {
entry:
  ; At x = 0.5, loop runs 4 steps (0.5^4 = 0.0625 < 0.1), triggering both
  ; initial allocation (iv = 0, size = 1) and buffer replacement + free
  ; (iv = 2, size = 3); d/dx(x^4) = 4 * 0.5^3 = 0.5.
  %dx = call double @df(double 5.000000e-01)
  %err = fsub double %dx, 5.000000e-01
  %err_sq = fmul double %err, %err
  %bad = fcmp ugt double %err_sq, 1.000000e-24
  %rc = select i1 %bad, i32 1, i32 0
  ret i32 %rc
}
