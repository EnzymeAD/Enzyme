; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; The adjoint of a struct load reinterprets the struct's differential as a
; vector of its float fields, through a temporary alloca. Those allocas are
; only aligned like the struct, so the vector loads from them must not take
; the larger ABI alignment of the vector type.

; The leading i64 is an integer, so the floats start at offset 8 and are read
; through a packed struct (gep.alloca).
define double @tester_gep(ptr %p, ptr %q) {
entry:
  %v = load { i64, [2 x double] }, ptr %p, align 8
  %n = extractvalue { i64, [2 x double] } %v, 0
  %g = getelementptr inbounds double, ptr %q, i64 %n
  %x = load double, ptr %g, align 8
  %a = extractvalue { i64, [2 x double] } %v, 1, 0
  %b = extractvalue { i64, [2 x double] } %v, 1, 1
  %s = fadd double %a, %b
  %r = fmul double %s, %x
  ret double %r
}

; Here the whole struct is treated as floats and read as <3 x double>
; (cast.alloca).
define double @tester_cast(ptr %p) {
entry:
  %v = load { i64, [2 x double] }, ptr %p, align 8
  %a = extractvalue { i64, [2 x double] } %v, 1, 0
  %b = extractvalue { i64, [2 x double] } %v, 1, 1
  %r = fadd double %a, %b
  ret double %r
}

define void @test_derivative(ptr %p, ptr %dp, ptr %q) {
entry:
  call void (...) @__enzyme_autodiff(ptr @tester_gep, ptr %p, ptr %dp, metadata !"enzyme_const", ptr %q)
  call void (...) @__enzyme_autodiff(ptr @tester_cast, ptr %p, ptr %dp)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal void @diffetester_gep(
; CHECK: %gep.alloca = alloca <{ [8 x i8], <2 x double>, [0 x i8] }>, align 8
; CHECK: %gep.ptr = getelementptr inbounds <{ [8 x i8], <2 x double>, [0 x i8] }>, ptr %gep.alloca, i64 0, i32 1
; CHECK-NEXT: %gep.load = load <2 x double>, ptr %gep.ptr, align 8
; CHECK-NEXT: %[[l1:.+]] = load <2 x double>, ptr %{{.*}}, align 8
; CHECK-NEXT: %[[a1:.+]] = fadd fast <2 x double> %[[l1]], %gep.load
; CHECK-NEXT: store <2 x double> %[[a1]], ptr %{{.*}}, align 8

; CHECK: define internal void @diffetester_cast(
; CHECK: %cast.alloca = alloca { i64, [2 x double] }, align 8
; CHECK: %cast.load = load <3 x double>, ptr %cast.alloca, align 8
; CHECK-NEXT: %[[l2:.+]] = load <3 x double>, ptr %"p'", align 8
; CHECK-NEXT: %[[a2:.+]] = fadd fast <3 x double> %[[l2]], %cast.load
; CHECK-NEXT: store <3 x double> %[[a2]], ptr %"p'", align 8
