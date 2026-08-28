; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=runtime_caller -S -o /dev/null | FileCheck %s --check-prefix=RUNTIME

; Reduced from flang -O1 output for a type-bound procedure ("s%step(...)"),
; which lowers to: load the derived type descriptor, load its binding table,
; index the table by the bound procedure's constant slot, and inttoptr-call the
; i64 found there. A bound procedure therefore has no direct call site anywhere
; -- its address appears only as a `ptrtoint` inside the table -- leaving type
; analysis nothing but the callee body to work from.

%_QM__fortran_type_infoTbinding = type { { i64 }, { ptr, i64 } }

@_QMmEXnXinit = linkonce_odr constant [4 x i8] c"init"
@_QMmEXnXstep = linkonce_odr constant [4 x i8] c"step"

; Entries are { c_funptr, name-descriptor }, 24 bytes each, so slot 1 (`step`)
; sits at byte 24.
@_QMmEXvXsolver_t = linkonce_odr constant [2 x %_QM__fortran_type_infoTbinding] [
  %_QM__fortran_type_infoTbinding {
    { i64 } { i64 ptrtoint (ptr @_QMmPinit to i64) },
    { ptr, i64 } { ptr @_QMmEXnXinit, i64 4 } },
  %_QM__fortran_type_infoTbinding {
    { i64 } { i64 ptrtoint (ptr @_QMmPstep to i64) },
    { ptr, i64 } { ptr @_QMmEXnXstep, i64 4 } }
], align 64

; The derived type descriptor; its first member is the binding table.
@_QMmEXdtXsolver_t = linkonce_odr constant { ptr, i64 } { ptr @_QMmEXvXsolver_t, i64 2 }

declare void @use(ptr, ptr)

; That %rwork is a real(8) work array is visible only here.
define void @caller(ptr %rwork, ptr %n) {
entry:
  %d = load double, ptr %rwork, align 8
  %d2 = fadd double %d, 1.000000e+00
  store double %d2, ptr %rwork, align 8
  %vt = load ptr, ptr @_QMmEXdtXsolver_t, align 8
  %slot = getelementptr i8, ptr %vt, i64 24
  %fpi = load i64, ptr %slot, align 8
  %fp = inttoptr i64 %fpi to ptr
  call void %fp(ptr %rwork, ptr %n)
  ret void
}

; Nothing in this body says what %yh points at.
define void @_QMmPstep(ptr %yh, ptr %ldyh) {
entry:
  %len = load i64, ptr %ldyh, align 8
  %cmp = icmp sgt i64 %len, 0
  br i1 %cmp, label %body, label %exit

body:
  call void @use(ptr %yh, ptr %ldyh)
  br label %exit

exit:
  ret void
}

define void @_QMmPinit(ptr %a, ptr %b) {
entry:
  ret void
}

; Dispatch off a runtime class descriptor: which table %box reaches is unknown.
define void @runtime_caller(ptr %box, ptr %rwork, ptr %n) {
entry:
  %d = load double, ptr %rwork, align 8
  %d2 = fadd double %d, 1.000000e+00
  store double %d2, ptr %rwork, align 8
  %dt = load ptr, ptr %box, align 8
  %vt = load ptr, ptr %dt, align 8
  %slot = getelementptr i8, ptr %vt, i64 24
  %fpi = load i64, ptr %slot, align 8
  %fp = inttoptr i64 %fpi to ptr
  call void %fp(ptr %rwork, ptr %n)
  ret void
}

; @caller's dispatch resolves, so @_QMmPstep is analyzed interprocedurally at
; all, and %rwork's element type reaches its %yh.

; CHECK: caller - {} |
; CHECK-NEXT: ptr %rwork: {[-1]:Pointer, [-1,0]:Float@double}

; CHECK: _QMmPstep - {} |{[-1]:Pointer, [-1,0]:Float@double}:{} {[-1]:Pointer}:{}
; CHECK-NEXT: ptr %yh: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: ptr %ldyh: {[-1]:Pointer}

; @runtime_caller's does not, and nothing is assumed: no callee is analyzed,
; and no type flows back into %n.

; RUNTIME: runtime_caller - {} |
; RUNTIME-NEXT: ptr %box: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Pointer}
; RUNTIME-NEXT: ptr %rwork: {[-1]:Pointer, [-1,0]:Float@double}
; RUNTIME-NEXT: ptr %n: {[-1]:Pointer}
; RUNTIME-NOT: _QMmPstep -
