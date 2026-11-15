; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Swapping two doubles through i64 slots whose offset is computed with `or`
; (strength-reduced `4 * i + 2`) must still be recognized as a float swap and
; therefore swapped back in the reverse pass rather than mirrored forward.

@enzyme_dup = external global ptr

declare double @__enzyme_autodiff(...)

define double @f(ptr %d) {
  %i = or i64 0, 2
  %a = getelementptr [8 x i8], ptr %d, i64 %i
  %b = getelementptr [8 x i8], ptr %d, i64 4
  %va = load i64, ptr %a, align 8
  %vb = load i64, ptr %b, align 8
  store i64 %vb, ptr %a, align 8
  store i64 %va, ptr %b, align 8
  %p = getelementptr i8, ptr %d, i64 16
  %x = load double, ptr %p, align 8
  ret double %x
}

define void @df(ptr %d, ptr %dd) {
  call double (...) @__enzyme_autodiff(ptr @f, ptr @enzyme_dup, ptr %d, ptr %dd)
  ret void
}

; CHECK-LABEL: define internal void @diffef(ptr nocapture %d, ptr nocapture %"d'", double %differeturn)
; CHECK: %"a'ipg" = getelementptr [8 x i8], ptr %"d'", i64 %i
; CHECK: %"b'ipg" = getelementptr [8 x i8], ptr %"d'", i64 4
; CHECK-NOT: store i64 %{{.*}}, ptr %"a'ipg"
; CHECK-NOT: store i64 %{{.*}}, ptr %"b'ipg"
; CHECK: invert:
; CHECK: %[[B0:.+]] = load i64, ptr %"b'ipg"
; CHECK-NEXT: store i64 0, ptr %"b'ipg"
; CHECK: %[[A0:.+]] = load i64, ptr %"a'ipg"
; CHECK-NEXT: store i64 0, ptr %"a'ipg"
; CHECK: store double %{{.+}}, ptr %"b'ipg"
; CHECK: store double %{{.+}}, ptr %"a'ipg"
