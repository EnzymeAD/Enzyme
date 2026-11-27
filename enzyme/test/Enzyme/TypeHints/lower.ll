; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Enzyme replaces each __enzyme_typed_value call by its operand and moves the
; call's type annotation onto the operand before analyzing any function.

declare double @__enzyme_autodiff(ptr, ...)
declare i64 @__enzyme_typed_value.i64(i64)

@enzyme_const = global i32 0

define double @f(ptr %x, ptr %c) {
entry:
  %s = load i64, ptr %c, align 8
  %t = call i64 @__enzyme_typed_value.i64(i64 %s), !enzyme_type !0
  %off = mul i64 %t, 8
  %p = getelementptr i8, ptr %x, i64 %off
  %v = load double, ptr %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @df(ptr %x, ptr %dx, ptr %c) {
entry:
  %r = call double (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, ptr @enzyme_const, ptr %c)
  ret double %r
}

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Integer"}

; CHECK-NOT: __enzyme_typed_value
; CHECK-LABEL: define double @f(
; CHECK: %s = load i64, ptr %c, align 8, !enzyme_type ![[INT:[0-9]+]]
; CHECK-NEXT: %off = mul i64 %s, 8
; CHECK-LABEL: define internal void @diffef(
; CHECK: %s = load i64, ptr %c, align 8, {{.*}}!enzyme_type ![[INT]]
; CHECK-NEXT: %off = mul i64 %s, 8
; CHECK: ![[INT]] = !{!"Unknown", i32 -1, ![[INT1:[0-9]+]]}
; CHECK: ![[INT1]] = !{!"Integer"}
