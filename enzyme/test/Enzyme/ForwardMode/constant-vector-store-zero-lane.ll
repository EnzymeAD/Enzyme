; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; A constant vector with a zero lane is stored to memory that nothing else
; types at that lane; the zero lane takes the type of the stored value.

define double @f(double %x) {
  %buf = call noalias ptr @malloc(i64 32)
  store double %x, ptr %buf, align 8
  %hi = getelementptr inbounds i8, ptr %buf, i64 16
  store <2 x double> <double 1.000000e+01, double 0.000000e+00>, ptr %hi, align 8
  %v = load double, ptr %buf, align 8
  %w = load double, ptr %hi, align 8
  %r = fmul double %v, %w
  call void @free(ptr %buf)
  ret double %r
}

define double @df(double %x) {
  %r = call double (...) @__enzyme_fwddiff(ptr @f, double %x, double 1.000000e+00)
  ret double %r
}

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffef(double %x, double %"x'")
; CHECK: store <2 x double> %{{.+}}, ptr %"hi'ipg", align 8
; CHECK: ret double
