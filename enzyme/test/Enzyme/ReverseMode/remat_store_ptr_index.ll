; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -O2 | %lli - ; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,default<O2>" | %lli -

declare void @__enzyme_autodiff(...)
declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare void @abort()

define void @fill_index(ptr nocapture %buffer, i64 %index) {
entry:
  store i64 %index, ptr %buffer, align 8
  ret void
}

define double @remat_malloc_index(ptr %values, i64 %index) {
entry:
  %index_buffer = call noalias ptr @malloc(i64 16)
  call void @fill_index(ptr %index_buffer, i64 %index)
  %remat = call noalias ptr @malloc(i64 32)
  store i64 0, ptr %remat, align 8
  %zero_gep = getelementptr inbounds i64, ptr %index_buffer, i64 0
  %store_index = load i64, ptr %zero_gep, align 8
  %store_gep = getelementptr inbounds i64, ptr %remat, i64 %store_index
  store i64 2, ptr %store_gep, align 8
  %load_gep = getelementptr inbounds i64, ptr %remat, i64 1
  %element = load i64, ptr %load_gep, align 8
  %value_gep = getelementptr inbounds double, ptr %values, i64 %element
  %value = load double, ptr %value_gep, align 8
  %square = fmul double %value, %value
  call void @free(ptr %index_buffer)
  call void @free(ptr %remat)
  ret double %square
}

define double @remat_shared_gep(ptr %values, ptr %index_ptr) {
entry:
  %remat = call noalias ptr @malloc(i64 32)
  %index = load i64, ptr %index_ptr, align 8
  store i64 0, ptr %index_ptr, align 8
  %shared_gep = getelementptr inbounds i64, ptr %remat, i64 %index
  store i64 3, ptr %shared_gep, align 8
  %element = load i64, ptr %shared_gep, align 8
  %value_gep = getelementptr inbounds double, ptr %values, i64 %element
  %value = load double, ptr %value_gep, align 8
  %square = fmul double %value, %value
  call void @free(ptr %remat)
  ret double %square
}

define double @outer(ptr %values, i64 %index, ptr %index_ptr) {
entry:
  %first = call double @remat_malloc_index(ptr %values, i64 %index)
  %second = call double @remat_shared_gep(ptr %values, ptr %index_ptr)
  %prod = fmul double %first, %second
  ret double %prod
}

define i32 @main() {
entry:
  %values = alloca [4 x double], align 8
  %dvalues = alloca [4 x double], align 8
  %index_slot = alloca i64, align 8
  %v0 = getelementptr inbounds [4 x double], ptr %values, i64 0, i64 0
  %v1 = getelementptr inbounds [4 x double], ptr %values, i64 0, i64 1
  %v2 = getelementptr inbounds [4 x double], ptr %values, i64 0, i64 2
  %v3 = getelementptr inbounds [4 x double], ptr %values, i64 0, i64 3
  store double 3.000000e+00, ptr %v0, align 8
  store double 5.000000e+00, ptr %v1, align 8
  store double 7.000000e+00, ptr %v2, align 8
  store double 1.100000e+01, ptr %v3, align 8
  %dv0 = getelementptr inbounds [4 x double], ptr %dvalues, i64 0, i64 0
  %dv1 = getelementptr inbounds [4 x double], ptr %dvalues, i64 0, i64 1
  %dv2 = getelementptr inbounds [4 x double], ptr %dvalues, i64 0, i64 2
  %dv3 = getelementptr inbounds [4 x double], ptr %dvalues, i64 0, i64 3
  store double 0.000000e+00, ptr %dv0, align 8
  store double 0.000000e+00, ptr %dv1, align 8
  store double 0.000000e+00, ptr %dv2, align 8
  store double 0.000000e+00, ptr %dv3, align 8
  store i64 2, ptr %index_slot, align 8
  call void (...) @__enzyme_autodiff(ptr @outer, metadata !"enzyme_dup", ptr %values, ptr %dvalues, metadata !"enzyme_const", i64 1, metadata !"enzyme_const", ptr %index_slot)
  %grad0 = load double, ptr %dv0, align 8
  %ok0 = fcmp oeq double %grad0, 0.000000e+00
  br i1 %ok0, label %check2, label %fail

check2:
  ; prod = (7^2) * (11^2) = 49 * 121
  ; d(prod)/d(values[2]) = 2 * 7 * 121 = 1694
  %grad2 = load double, ptr %dv2, align 8
  %ok2 = fcmp oeq double %grad2, 1.694000e+03
  br i1 %ok2, label %check3, label %fail

check3:
  ; d(prod)/d(values[3]) = 49 * 2 * 11 = 1078
  %grad3 = load double, ptr %dv3, align 8
  %ok3 = fcmp oeq double %grad3, 1.078000e+03
  br i1 %ok3, label %done, label %fail

fail:
  call void @abort()
  unreachable

done:
  ret i32 0
}

; CHECK-LABEL: define internal void @differemat_shared_gep(
; CHECK-SAME: i64 %[[CACHED_INDEX:.+]])
; CHECK:   %remat = call noalias ptr @malloc(i64 32)
; CHECK:   %shared_gep = getelementptr inbounds i64, ptr %remat, i64 %[[CACHED_INDEX]]
; CHECK:   store i64 3, ptr %shared_gep, align 8

; CHECK-LABEL: define internal void @differemat_malloc_index(
; CHECK:   %index_buffer = extractvalue { ptr, double } %tapeArg, 0
; CHECK:   %remat = call noalias ptr @malloc(i64 32)
; CHECK:   %store_gep = getelementptr inbounds i64, ptr %remat, i64 %store_index
; CHECK:   store i64 2, ptr %store_gep, align 8
