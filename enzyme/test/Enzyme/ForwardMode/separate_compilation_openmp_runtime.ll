; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-global-activity -enzyme-globals-default-inactive -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then rm -rf %t && mkdir -p %t && %opt < %s %OPnewLoadEnzyme -passes="enzyme-summary" -enzyme-summary-out=%t/m.json -disable-output && python3 %S/../../../scripts/enzyme_thinlink.py --inactive both --out %t/plan %t/m.json | FileCheck %s --check-prefix=REPORT; fi

; Separate compilation: the OpenMP runtime queries, in the C and the Fortran
; (omp_lib without BIND(C)) calling conventions, are inactive and have no
; derivative in another module, like the flang runtime. ICON's
; mo_nh_nest_utilities calls omp_get_max_threads_; the link then failed on
; the undefined __enzyme_sep_fwd_w1_omp_get_max_threads_.

declare i32 @omp_get_max_threads_()
declare i32 @omp_get_thread_num()

define void @f(ptr %x, i64 %n) {
entry:
  %t = call i32 @omp_get_max_threads_()
  %k = call i32 @omp_get_thread_num()
  %s = add i32 %t, %k
  %sd = sitofp i32 %s to double
  %v = load double, ptr %x
  %m = fmul double %v, %sd
  store double %m, ptr %x
  ret void
}

declare void @__enzyme_fwddiff(...)

define void @caller(ptr %x, ptr %dx, i64 %n) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @f, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i64 %n)
  ret void
}

; CHECK-NOT: __enzyme_sep_{{.*}}omp_
; CHECK: define internal void @fwddiffef(
; CHECK:   %t = call i32 @omp_get_max_threads_()
; CHECK:   %k = call i32 @omp_get_thread_num()

; REPORT-NOT: omp_get_max_threads_
; REPORT-NOT: omp_get_thread_num
