; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

define <2 x double> @f_vec(<2 x double> %x) {
  %r = fmul <2 x double> %x, %x
  ret <2 x double> %r
}

define double @f_scalar(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

declare { double, double, double, double } @__enzyme_fwddiff_flat(...)
declare { <2 x double>, <2 x double> } @__enzyme_fwddiff_vec2(...)
declare void @__enzyme_fwddiff_sret(ptr sret({ double, double, double, double }), ...)
declare { double, [3 x double] } @__enzyme_fwddiff_primal_arr(...)
declare { double, { double, double, double } } @__enzyme_fwddiff_primal_st(...)

define { double, double, double, double } @test_flat(<2 x double> %x, <2 x double> %dx1, <2 x double> %dx2) {
  %r = call { double, double, double, double } (...) @__enzyme_fwddiff_flat(ptr @f_vec, metadata !"enzyme_width", i64 2, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2)
  ret { double, double, double, double } %r
}

define { <2 x double>, <2 x double> } @test_vec2(<2 x double> %x, <2 x double> %dx1, <2 x double> %dx2) {
  %r = call { <2 x double>, <2 x double> } (...) @__enzyme_fwddiff_vec2(ptr @f_vec, metadata !"enzyme_width", i64 2, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2)
  ret { <2 x double>, <2 x double> } %r
}

define void @test_sret(ptr %out, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2) {
  call void (ptr, ...) @__enzyme_fwddiff_sret(ptr sret({ double, double, double, double }) %out, ptr @f_vec, metadata !"enzyme_width", i64 2, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2)
  ret void
}

define { double, [3 x double] } @test_primal_arr(double %x, double %dx1, double %dx2, double %dx3) {
  %r = call { double, [3 x double] } (...) @__enzyme_fwddiff_primal_arr(ptr @f_scalar, metadata !"enzyme_width", i64 3, metadata !"enzyme_primal_return", double %x, double %dx1, double %dx2, double %dx3)
  ret { double, [3 x double] } %r
}

define { double, { double, double, double } } @test_primal_st(double %x, double %dx1, double %dx2, double %dx3) {
  %r = call { double, { double, double, double } } (...) @__enzyme_fwddiff_primal_st(ptr @f_scalar, metadata !"enzyme_width", i64 3, metadata !"enzyme_primal_return", double %x, double %dx1, double %dx2, double %dx3)
  ret { double, { double, double, double } } %r
}

; CHECK: define { double, double, double, double } @test_flat(
; CHECK:   %[[CALL1:.+]] = call fast [2 x <2 x double>] @fwddiffe2f_vec(
; CHECK-NEXT:   %[[V0:.+]] = extractvalue [2 x <2 x double>] %[[CALL1]], 0
; CHECK-NEXT:   %[[E0:.+]] = extractelement <2 x double> %[[V0]], i64 0
; CHECK-NEXT:   %[[A0:.+]] = insertvalue { double, double, double, double } zeroinitializer, double %[[E0]], 0
; CHECK-NEXT:   %[[E1:.+]] = extractelement <2 x double> %[[V0]], i64 1
; CHECK-NEXT:   %[[A1:.+]] = insertvalue { double, double, double, double } %[[A0]], double %[[E1]], 1
; CHECK-NEXT:   %[[V1:.+]] = extractvalue [2 x <2 x double>] %[[CALL1]], 1
; CHECK-NEXT:   %[[E2:.+]] = extractelement <2 x double> %[[V1]], i64 0
; CHECK-NEXT:   %[[A2:.+]] = insertvalue { double, double, double, double } %[[A1]], double %[[E2]], 2
; CHECK-NEXT:   %[[E3:.+]] = extractelement <2 x double> %[[V1]], i64 1
; CHECK-NEXT:   %[[A3:.+]] = insertvalue { double, double, double, double } %[[A2]], double %[[E3]], 3
; CHECK-NEXT:   ret { double, double, double, double } %[[A3]]

; CHECK: define { <2 x double>, <2 x double> } @test_vec2(
; CHECK:   %[[CALL2:.+]] = call fast [2 x <2 x double>] @fwddiffe2f_vec(
; CHECK-NEXT:   %[[W0:.+]] = extractvalue [2 x <2 x double>] %[[CALL2]], 0
; CHECK-NEXT:   %[[B0:.+]] = insertvalue { <2 x double>, <2 x double> } zeroinitializer, <2 x double> %[[W0]], 0
; CHECK-NEXT:   %[[W1:.+]] = extractvalue [2 x <2 x double>] %[[CALL2]], 1
; CHECK-NEXT:   %[[B1:.+]] = insertvalue { <2 x double>, <2 x double> } %[[B0]], <2 x double> %[[W1]], 1
; CHECK-NEXT:   ret { <2 x double>, <2 x double> } %[[B1]]

; CHECK: define void @test_sret(
; CHECK:   %[[CALL3:.+]] = call fast [2 x <2 x double>] @fwddiffe2f_vec(
; CHECK:   insertvalue { double, double, double, double } zeroinitializer, double {{%.+}}, 0
; CHECK:   insertvalue { double, double, double, double } {{%.+}}, double {{%.+}}, 1
; CHECK:   insertvalue { double, double, double, double } {{%.+}}, double {{%.+}}, 2
; CHECK:   %[[S3:.+]] = insertvalue { double, double, double, double } {{%.+}}, double {{%.+}}, 3
; CHECK-NEXT:   store { double, double, double, double } %[[S3]], ptr %out

; CHECK: define { double, [3 x double] } @test_primal_arr(
; CHECK:   %[[CALL4:.+]] = call { double, [3 x double] } @fwddiffe3f_scalar(
; CHECK-NEXT:   ret { double, [3 x double] } %[[CALL4]]

; CHECK: define { double, { double, double, double } } @test_primal_st(
; CHECK:   %[[CALL5:.+]] = call { double, [3 x double] } @fwddiffe3f_scalar(
; CHECK:   insertvalue { double, double, double } zeroinitializer, double {{%.+}}, 0
; CHECK:   insertvalue { double, double, double } {{%.+}}, double {{%.+}}, 1
; CHECK:   insertvalue { double, double, double } {{%.+}}, double {{%.+}}, 2
; CHECK:   ret { double, { double, double, double } }
