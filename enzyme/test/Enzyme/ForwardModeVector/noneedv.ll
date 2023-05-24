; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

source_filename = "<source>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local void @in(double* noalias %a, double* noalias %b) {
entry:
  %ld = load double, double* %a, align 8
  store double %ld, double* %b, align 8
  ret void
}

; Function Attrs: noinline optnone uwtable mustprogress
define dso_local void @f(double* noalias %a, double* noalias %b, double* noalias %c, double* noalias %d) #1 {
  call void (...) @_Z16__enzyme_fwddiffIvET_Pvz(void (double*, double*)* @in, metadata !"enzyme_dup", double* %a, double* %b, metadata !"enzyme_dupnoneedv", i64 80, double* %c, double* %d)
  ret void
}

declare dso_local void @_Z16__enzyme_fwddiffIvET_Pvz(...)

; CHECK: define internal void @fwddiffein(double* noalias %a, double* %"a'", double* noalias %b, double* %"b'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"ld'ipl" = load double, double* %"a'"
; CHECK-NEXT:   store double %"ld'ipl", double* %"b'"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
