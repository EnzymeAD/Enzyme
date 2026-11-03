; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s; fi

; aligned_alloc(alignment, size) is an allocation function, freed by free;
; its shadow is allocated like the primal (e.g. flang's array temporaries).

declare void @__enzyme_fwddiff(...)

define void @caller(double %x, ptr %out, ptr %dout) {
  call void (...) @__enzyme_fwddiff(ptr @tmp, double %x, double 1.0, ptr %out, ptr %dout)
  ret void
}

define void @tmp(double %x, ptr %out) {
  %buf = call noalias ptr @aligned_alloc(i64 64, i64 128)
  store double %x, ptr %buf, align 8
  %v = load double, ptr %buf, align 8
  %m = fmul double %v, %v
  store double %m, ptr %out, align 8
  call void @free(ptr %buf)
  ret void
}

declare ptr @aligned_alloc(i64, i64)

declare void @free(ptr)

; CHECK-LABEL: define internal void @fwddiffetmp(
; CHECK-DAG: %buf = call noalias ptr @aligned_alloc(i64 64, i64 128)
; CHECK-DAG: %[[dbuf:[^ ]+]] = call {{.*}}ptr @aligned_alloc(i64 64, i64 128)
; CHECK-DAG: store double %"x'", ptr %[[dbuf]]
; CHECK-DAG: call void @free(ptr {{.*}}%[[dbuf]])
; CHECK-DAG: call void @free(ptr {{.*}}%buf)
