; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify)" -S | FileCheck %s; fi

%struct.View = type { ptr, i64 }

declare void @__enzyme_autodiff(...)

define internal void @scale_view(ptr %data, i64 %len, double %factor) {
entry:
  %v = load double, ptr %data, align 8
  %mul = fmul double %v, %factor
  store double %mul, ptr %data, align 8
  ret void
}

; Case 1: Both primal and shadow spilled to ptr byval(%struct.View)
define double @test_both_spilled_byval(ptr %x, ptr %dx, double %factor) {
entry:
  %v = alloca %struct.View, align 8
  %dv = alloca %struct.View, align 8
  %v_p = getelementptr inbounds %struct.View, ptr %v, i64 0, i32 0
  %v_n = getelementptr inbounds %struct.View, ptr %v, i64 0, i32 1
  store ptr %x, ptr %v_p, align 8
  store i64 1, ptr %v_n, align 8
  %dv_p = getelementptr inbounds %struct.View, ptr %dv, i64 0, i32 0
  %dv_n = getelementptr inbounds %struct.View, ptr %dv, i64 0, i32 1
  store ptr %dx, ptr %dv_p, align 8
  store i64 1, ptr %dv_n, align 8
  %res = call double (...) @__enzyme_autodiff(ptr @scale_view, metadata !"enzyme_dup", ptr byval(%struct.View) align 8 %v, ptr byval(%struct.View) align 8 %dv, metadata !"enzyme_out", double %factor)
  ret double %res
}

; Case 2: Primal passed in coerced registers (ptr, i64), shadow spilled to ptr byval(%struct.View)
define double @test_reg_primal_byval_shadow(ptr %x, ptr %dx, double %factor) {
entry:
  %dv = alloca %struct.View, align 8
  %dv_p = getelementptr inbounds %struct.View, ptr %dv, i64 0, i32 0
  %dv_n = getelementptr inbounds %struct.View, ptr %dv, i64 0, i32 1
  store ptr %dx, ptr %dv_p, align 8
  store i64 1, ptr %dv_n, align 8
  %res = call double (...) @__enzyme_autodiff(ptr @scale_view, metadata !"enzyme_dup", ptr %x, i64 1, ptr byval(%struct.View) align 8 %dv, metadata !"enzyme_out", double %factor)
  ret double %res
}

; Case 3: Single-pointer struct coerced as i64 at callsite while callee expects ptr
define double @test_i64_coerced_ptr(ptr %x, ptr %dx, double %factor) {
entry:
  %xi = ptrtoint ptr %x to i64
  %dxi = ptrtoint ptr %dx to i64
  %res = call double (...) @__enzyme_autodiff(ptr @scale_view, metadata !"enzyme_dup", i64 %xi, i64 %dxi, metadata !"enzyme_const", i64 1, metadata !"enzyme_out", double %factor)
  ret double %res
}

; CHECK-LABEL: define double @test_both_spilled_byval(
; CHECK: call {{.*}}@diffescale_view(ptr {{.*}}, ptr {{.*}}, i64 {{.*}}, i64 {{.*}}, double %factor)
; CHECK: ret double

; CHECK-LABEL: define double @test_reg_primal_byval_shadow(
; CHECK: call {{.*}}@diffescale_view(ptr %x, ptr {{.*}}, i64 1, i64 {{.*}}, double %factor)
; CHECK: ret double

; CHECK-LABEL: define double @test_i64_coerced_ptr(
; CHECK: call {{.*}}@diffescale_view{{.*}}(ptr %x, ptr %dx, i64 1, double %factor)
; CHECK: ret double
