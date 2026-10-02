; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -S | FileCheck %s --check-prefix=USE; fi
; RUN: if [ %llvmver -ge 16 ]; then printf "ext reverse+aa\n" > %t.exports; %opt < %S/Inputs/separate_compilation_omp_ext.ll.in %OPnewLoadEnzyme -passes="enzyme" -enzyme-separate-compilation -enzyme-export-list=%t.exports -S | FileCheck %s --check-prefix=DEF; fi

; Separate compilation: a call through another module's derivative from an
; OpenMP parallel region accumulates adjoints atomically, which is part of
; the derivative's name (_aa). The defining module exports that variant on
; request ("+aa" in -enzyme-export-list).

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

declare !callback !0 void @__kmpc_fork_call(ptr, i32, ptr, ...)
declare void @ext(ptr, ptr)
declare void @__enzyme_autodiff(...)

define internal void @outlined(ptr noalias %gtid, ptr noalias %btid, ptr %x, ptr %y) {
  %t = load i32, ptr %gtid
  %i = sext i32 %t to i64
  %yi = getelementptr inbounds double, ptr %y, i64 %i
  call void @ext(ptr %x, ptr %yi)
  ret void
}

define void @f(ptr %x, ptr %y) {
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 2, ptr @outlined, ptr %x, ptr %y)
  ret void
}

define void @caller(ptr %x, ptr %dx, ptr %y, ptr %dy) {
  call void (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, ptr %y, ptr %dy)
  ret void
}

!0 = !{!1}
!1 = !{i64 2, i64 -1, i64 -1, i1 true}

; USE: @__enzyme_sep_rev_w1_aa_ext = external constant { ptr, ptr }

; DEF: @__enzyme_sep_rev_w1_aa_ext = constant { ptr, ptr } { ptr @augmented_ext, ptr @diffeext }
; DEF: define internal void @diffeext(
; DEF: atomicrmw fadd
