; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=kernel -o /dev/null | FileCheck %s --check-prefix=TA-INT
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=float_atomic_kernel -o /dev/null | FileCheck %s --check-prefix=TA-FLT
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare i64 @__atomic_load_8(ptr, i32) nounwind
declare void @__atomic_store_8(ptr, i64, i32) nounwind
declare zeroext i1 @__atomic_compare_exchange_8(ptr, ptr, i64, i1, i32, i32) nounwind
declare double @__enzyme_autodiff(...)

define double @kernel(double %x, ptr %atom) {
entry:
  %expected = alloca i64, align 8
  %v0 = call i64 @__atomic_load_8(ptr %atom, i32 0)
  store i64 %v0, ptr %expected, align 8
  %ok = call zeroext i1 @__atomic_compare_exchange_8(ptr %atom, ptr %expected, i64 1, i1 false, i32 0, i32 0)
  call void @__atomic_store_8(ptr %atom, i64 2, i32 0)
  %scale = select i1 %ok, double 3.000000e+00, double 2.000000e+00
  %res = fmul double %x, %scale
  ret double %res
}

define double @float_atomic_kernel(ptr %atom, double %x) {
entry:
  %expected = alloca double, align 8
  %v0_bits = call i64 @__atomic_load_8(ptr %atom, i32 0)
  %v0 = bitcast i64 %v0_bits to double
  store double %v0, ptr %expected, align 8
  %des = fadd double %v0, %x
  %des_bits = bitcast double %des to i64
  %ok = call zeroext i1 @__atomic_compare_exchange_8(ptr %atom, ptr %expected, i64 %des_bits, i1 false, i32 0, i32 0)
  call void @__atomic_store_8(ptr %atom, i64 %des_bits, i32 0)
  ret double %des
}

define double @grad_kernel(double %x, ptr %atom) {
entry:
  %dx = call double (...) @__enzyme_autodiff(ptr @kernel, double %x, metadata !"enzyme_const", ptr %atom)
  ret double %dx
}

; TA-INT-LABEL: kernel -
; TA-INT: ptr %atom: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; TA-INT: %expected = alloca i64, align 8: {[-1]:Pointer, [-1,-1]:Integer}
; TA-INT: %v0 = call i64 @__atomic_load_8(ptr %atom, i32 0): {[-1]:Integer}
; TA-INT: %ok = call zeroext i1 @__atomic_compare_exchange_8(ptr %atom, ptr %expected, i64 1, i1 false, i32 0, i32 0): {[-1]:Integer}

; TA-FLT-LABEL: float_atomic_kernel -
; TA-FLT: ptr %atom: {[-1]:Pointer, [-1,0]:Float@double}
; TA-FLT: %expected = alloca double, align 8: {[-1]:Pointer, [-1,-1]:Float@double}
; TA-FLT: %v0_bits = call i64 @__atomic_load_8(ptr %atom, i32 0): {[-1]:Float@double}
; TA-FLT: %ok = call zeroext i1 @__atomic_compare_exchange_8(ptr %atom, ptr %expected, i64 %des_bits, i1 false, i32 0, i32 0): {[-1]:Integer}

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK-NOT: NoShadowError
; CHECK-NOT: NoDerivativeError
; CHECK: call i64 @__atomic_load_8(
; CHECK: call zeroext i1 @__atomic_compare_exchange_8(
; CHECK: call void @__atomic_store_8(
; CHECK: ret { double }
