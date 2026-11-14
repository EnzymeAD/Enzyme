; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

define <2 x double> @f_vec(<2 x double> %x) {
  %r = fmul <2 x double> %x, %x
  ret <2 x double> %r
}

declare { double, double, double, double } @__enzyme_batch_flat(...)
declare { <2 x double>, <2 x double> } @__enzyme_batch_vec2(...)
declare void @__enzyme_batch_sret(ptr sret({ double, double, double, double }), ...)

define { double, double, double, double } @test_flat(<2 x double> %x1, <2 x double> %x2) {
  %r = call { double, double, double, double } (...) @__enzyme_batch_flat(ptr @f_vec, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", <2 x double> %x1, <2 x double> %x2)
  ret { double, double, double, double } %r
}

define { <2 x double>, <2 x double> } @test_vec2(<2 x double> %x1, <2 x double> %x2) {
  %r = call { <2 x double>, <2 x double> } (...) @__enzyme_batch_vec2(ptr @f_vec, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", <2 x double> %x1, <2 x double> %x2)
  ret { <2 x double>, <2 x double> } %r
}

define void @test_sret(ptr %out, <2 x double> %x1, <2 x double> %x2) {
  call void (ptr, ...) @__enzyme_batch_sret(ptr sret({ double, double, double, double }) %out, ptr @f_vec, metadata !"enzyme_width", i64 2, metadata !"enzyme_vector", <2 x double> %x1, <2 x double> %x2)
  ret void
}

; CHECK: define { double, double, double, double } @test_flat(
; CHECK:   %[[CALL1:.+]] = call [2 x <2 x double>] @batch_f_vec(
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
; CHECK:   %[[CALL2:.+]] = call [2 x <2 x double>] @batch_f_vec(
; CHECK-NEXT:   %[[W0:.+]] = extractvalue [2 x <2 x double>] %[[CALL2]], 0
; CHECK-NEXT:   %[[B0:.+]] = insertvalue { <2 x double>, <2 x double> } zeroinitializer, <2 x double> %[[W0]], 0
; CHECK-NEXT:   %[[W1:.+]] = extractvalue [2 x <2 x double>] %[[CALL2]], 1
; CHECK-NEXT:   %[[B1:.+]] = insertvalue { <2 x double>, <2 x double> } %[[B0]], <2 x double> %[[W1]], 1
; CHECK-NEXT:   ret { <2 x double>, <2 x double> } %[[B1]]

; CHECK: define void @test_sret(
; CHECK:   %[[CALL3:.+]] = call [2 x <2 x double>] @batch_f_vec(
; CHECK:   insertvalue { double, double, double, double } zeroinitializer, double {{%.+}}, 0
; CHECK:   insertvalue { double, double, double, double } {{%.+}}, double {{%.+}}, 1
; CHECK:   insertvalue { double, double, double, double } {{%.+}}, double {{%.+}}, 2
; CHECK:   %[[S3:.+]] = insertvalue { double, double, double, double } {{%.+}}, double {{%.+}}, 3
; CHECK-NEXT:   store { double, double, double, double } %[[S3]], ptr %out
