; RUN: if [ %llvmver -ge 16 ]; then not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -disable-output 2>&1 | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then not %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -disable-output 2>&1 | FileCheck %s; fi

define double @f(double %x) {
  %d = call double (...) @__enzyme_fwddiff(ptr @f, double %x, double 1.0)
  ret double %d
}

declare double @__enzyme_fwddiff(...)

; CHECK: Enzyme: No forward mode derivative found for __enzyme_fwddiff
; CHECK-NEXT: at context: {{.*}}@__enzyme_fwddiff(ptr @f,
