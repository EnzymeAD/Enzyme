; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S 2>&1 | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S 2>&1 | FileCheck %s

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare double @__enzyme_autodiff(ptr, ...)

define double @objective(double %x, i1 %cond, i1 %cond2) {
entry:
  br i1 %cond, label %alloc.bb, label %null.bb

null.bb:
  br label %cleanup

alloc.bb:
  %p = call noalias nonnull dereferenceable(16) ptr @malloc(i64 16)
  store double %x, ptr %p, align 8
  %p1_init = getelementptr inbounds double, ptr %p, i64 1
  store double 0.000000e+00, ptr %p1_init, align 8
  br i1 %cond2, label %extra.bb, label %cleanup

extra.bb:
  br label %cleanup

cleanup:
  %phi = phi ptr [ %p, %alloc.bb ], [ null, %null.bb ], [ %p, %extra.bb ]
  %not_null = icmp ne ptr %phi, null
  br i1 %not_null, label %use.bb, label %exit.bb

use.bb:
  %v0 = load double, ptr %phi, align 8
  %p1 = getelementptr inbounds double, ptr %phi, i64 1
  %v1 = load double, ptr %p1, align 8
  %sum = fadd double %v0, %v1
  br label %exit.bb

exit.bb:
  %res = phi double [ %sum, %use.bb ], [ 0.000000e+00, %cleanup ]
  call void @free(ptr %phi)
  ret double %res
}

define double @caller(double %x, i1 %cond, i1 %cond2) {
entry:
  %d = call double (ptr, ...) @__enzyme_autodiff(ptr @objective, double %x, i1 %cond, i1 %cond2)
  ret double %d
}

; CHECK-NOT: freeing without malloc
; CHECK: define internal { double } @diffeobjective(double %x, i1 %cond, i1 %cond2, double %differeturn)
; CHECK: invertalloc.bb:
; CHECK:   call void @free(ptr nonnull %{{.+}})
; CHECK:   call void @free(ptr %{{.+}})
