; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S 2>&1 | FileCheck %s --check-prefix=QUIET
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -pass-remarks=enzyme -S 2>&1 | FileCheck %s --check-prefix=REMARK

; Verify that removing a free without a matching malloc or a free of null
; emits optimization remarks via EmitWarning rather than writing unconditionally
; to stderr.

declare void @free(ptr)
declare double @__enzyme_autodiff(...)

define double @free_unknown(ptr %p, double %x) {
  %ld = load ptr, ptr %p
  call void @free(ptr %ld)
  %sq = fmul double %x, %x
  ret double %sq
}

define double @free_null(double %x) {
  call void @free(ptr null)
  %sq = fmul double %x, %x
  ret double %sq
}

define double @caller(ptr %p, double %x) {
  %r1 = call double (...) @__enzyme_autodiff(ptr @free_unknown, metadata !"enzyme_const", ptr %p, double %x)
  %r2 = call double (...) @__enzyme_autodiff(ptr @free_null, double %x)
  %r = fadd double %r1, %r2
  ret double %r
}

; QUIET-NOT: freeing without malloc
; QUIET-NOT: removing free of null pointer
; QUIET: define internal { double } @diffefree_unknown
; QUIET: define internal { double } @diffefree_null

; REMARK: remark: {{.*}}: freeing without malloc {{.*}}%ld = load ptr, ptr %p
; REMARK: remark: {{.*}}: removing free of null pointer
