; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg),verify" -S | FileCheck %s; fi

declare ptr @gsl_fft_real_wavetable_alloc(i64)
declare void @gsl_fft_real_wavetable_free(ptr)
declare ptr @gsl_fft_halfcomplex_wavetable_alloc(i64)
declare void @gsl_fft_halfcomplex_wavetable_free(ptr)
declare ptr @gsl_fft_real_workspace_alloc(i64)
declare void @gsl_fft_real_workspace_free(ptr)
declare ptr @gsl_fft_complex_wavetable_alloc(i64)
declare void @gsl_fft_complex_wavetable_free(ptr)
declare ptr @gsl_fft_complex_workspace_alloc(i64)
declare void @gsl_fft_complex_workspace_free(ptr)

declare i32 @gsl_fft_real_transform(ptr, i64, i64, ptr, ptr)
declare i32 @gsl_fft_halfcomplex_backward(ptr, i64, i64, ptr, ptr)
declare i32 @gsl_fft_halfcomplex_inverse(ptr, i64, i64, ptr, ptr)
declare i32 @gsl_fft_real_radix2_transform(ptr, i64, i64)
declare i32 @gsl_fft_halfcomplex_radix2_inverse(ptr, i64, i64)
declare i32 @gsl_fft_complex_forward(ptr, i64, i64, ptr, ptr)
declare i32 @gsl_fft_complex_inverse(ptr, i64, i64, ptr, ptr)
declare i32 @gsl_fft_complex_transform(ptr, i64, i64, ptr, ptr, i32)
declare i32 @gsl_fft_halfcomplex_unpack(ptr, ptr, i64, i64)
declare i32 @gsl_fft_halfcomplex_radix2_unpack(ptr, ptr, i64, i64)
declare i32 @gsl_fft_real_unpack(ptr, ptr, i64, i64)

declare double @gsl_stats_mean(ptr, i64, i64)
declare double @gsl_stats_sd_m(ptr, i64, i64, double)
declare i64 @gsl_stats_max_index(ptr, i64, i64)

declare void @__enzyme_autodiff(...)

define void @real_fft_fn(ptr %data, i64 %stride, i64 %n) {
entry:
  %wt = call ptr @gsl_fft_real_wavetable_alloc(i64 %n)
  %wk = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
  %r = call i32 @gsl_fft_real_transform(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk)
  call void @gsl_fft_real_wavetable_free(ptr %wt)
  call void @gsl_fft_real_workspace_free(ptr %wk)
  ret void
}

define void @hc_inv_fn(ptr %data, i64 %stride, i64 %n) {
entry:
  %wt = call ptr @gsl_fft_halfcomplex_wavetable_alloc(i64 %n)
  %wk = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
  %r = call i32 @gsl_fft_halfcomplex_inverse(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk)
  call void @gsl_fft_halfcomplex_wavetable_free(ptr %wt)
  call void @gsl_fft_real_workspace_free(ptr %wk)
  ret void
}

define void @radix2_fn(ptr %data, ptr %cdata, i64 %stride, i64 %n) {
entry:
  %r0 = call i32 @gsl_fft_real_radix2_transform(ptr %data, i64 %stride, i64 %n)
  %r1 = call i32 @gsl_fft_halfcomplex_radix2_unpack(ptr %data, ptr %cdata, i64 %stride, i64 %n)
  %r2 = call i32 @gsl_fft_halfcomplex_radix2_inverse(ptr %data, i64 %stride, i64 %n)
  ret void
}

define void @complex_fn(ptr %data, i64 %stride, i64 %n, i32 %sign) {
entry:
  %wt = call ptr @gsl_fft_complex_wavetable_alloc(i64 %n)
  %wk = call ptr @gsl_fft_complex_workspace_alloc(i64 %n)
  %r0 = call i32 @gsl_fft_complex_forward(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk)
  %r1 = call i32 @gsl_fft_complex_inverse(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk)
  %r2 = call i32 @gsl_fft_complex_transform(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk, i32 %sign)
  call void @gsl_fft_complex_wavetable_free(ptr %wt)
  call void @gsl_fft_complex_workspace_free(ptr %wk)
  ret void
}

define double @demo_pipeline(ptr %data, ptr %cdata, i64 %n) {
entry:
  %wt = call ptr @gsl_fft_real_wavetable_alloc(i64 %n)
  %wk = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
  %r0 = call i32 @gsl_fft_real_transform(ptr %data, i64 1, i64 %n, ptr %wt, ptr %wk)
  call void @gsl_fft_real_wavetable_free(ptr %wt)
  call void @gsl_fft_real_workspace_free(ptr %wk)
  %r1 = call i32 @gsl_fft_halfcomplex_unpack(ptr %data, ptr %cdata, i64 1, i64 %n)
  %m = call double @gsl_stats_mean(ptr %cdata, i64 2, i64 %n)
  %sd = call double @gsl_stats_sd_m(ptr %cdata, i64 2, i64 %n, double %m)
  %idx = call i64 @gsl_stats_max_index(ptr %cdata, i64 2, i64 %n)
  %p = getelementptr inbounds double, ptr %cdata, i64 %idx
  %v = load double, ptr %p, align 8
  %sum0 = fadd double %m, %sd
  %sum1 = fadd double %sum0, %v
  ret double %sum1
}

define void @unpack_const_src(ptr %src, ptr %dst, i64 %stride, i64 %n) {
entry:
  %r0 = call i32 @gsl_fft_real_unpack(ptr %src, ptr %dst, i64 %stride, i64 %n)
  ret void
}

define void @caller(ptr %data, ptr %ddata, ptr %ddata2, ptr %cdata, ptr %dcdata, i64 %stride, i64 %n, i32 %sign) {
entry:
  call void (...) @__enzyme_autodiff(ptr @real_fft_fn, metadata !"enzyme_dup", ptr %data, ptr %ddata, i64 %stride, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @real_fft_fn, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %data, ptr %ddata, ptr %ddata2, i64 %stride, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @hc_inv_fn, metadata !"enzyme_dup", ptr %data, ptr %ddata, i64 %stride, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @radix2_fn, metadata !"enzyme_dup", ptr %data, ptr %ddata, metadata !"enzyme_dup", ptr %cdata, ptr %dcdata, i64 %stride, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @complex_fn, metadata !"enzyme_dup", ptr %data, ptr %ddata, i64 %stride, i64 %n, i32 %sign)
  call void (...) @__enzyme_autodiff(ptr @demo_pipeline, metadata !"enzyme_dup", ptr %data, ptr %ddata, metadata !"enzyme_dup", ptr %cdata, ptr %dcdata, i64 %n)
  call void (...) @__enzyme_autodiff(ptr @unpack_const_src, metadata !"enzyme_const", ptr %data, metadata !"enzyme_dup", ptr %cdata, ptr %dcdata, i64 %stride, i64 %n)
  ret void
}

; CHECK-LABEL: define internal void @differeal_fft_fn(ptr %data, ptr %"data'", i64 %stride, i64 %n)
; CHECK:   %wt = call ptr @gsl_fft_real_wavetable_alloc(i64 %n)
; CHECK:   %wk = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
; CHECK:   %r = call i32 @gsl_fft_real_transform(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk)
; CHECK:   call void @gsl_fft_real_wavetable_free(ptr %wt)
; CHECK:   call void @gsl_fft_real_workspace_free(ptr %wk)
; CHECK:   %[[HCWT:.+]] = call ptr @gsl_fft_halfcomplex_wavetable_alloc(i64 %n)
; CHECK:   %[[WK2:.+]] = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
; CHECK:   call void @__enzyme_gsl_fft_scale_hc_f64_i64(ptr %"data'", i64 %stride, i64 %n, double 5.000000e-01)
; CHECK:   %{{.+}} = call i32 @gsl_fft_halfcomplex_backward(ptr %"data'", i64 %stride, i64 %n, ptr %[[HCWT]], ptr %[[WK2]])
; CHECK:   call void @gsl_fft_halfcomplex_wavetable_free(ptr %[[HCWT]])
; CHECK:   call void @gsl_fft_real_workspace_free(ptr %[[WK2]])
; CHECK:   ret void

; CHECK-LABEL: define internal void @diffe2real_fft_fn(ptr %data, [2 x ptr] %"data'", i64 %stride, i64 %n)
; CHECK:   %[[VHCWT:.+]] = call ptr @gsl_fft_halfcomplex_wavetable_alloc(i64 %n)
; CHECK:   %[[VWK2:.+]] = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
; CHECK:   call void @__enzyme_gsl_fft_scale_hc_f64_i64(ptr {{.+}}, i64 %stride, i64 %n, double 5.000000e-01)
; CHECK:   call i32 @gsl_fft_halfcomplex_backward(ptr {{.+}}, i64 %stride, i64 %n, ptr %[[VHCWT]], ptr %[[VWK2]])
; CHECK:   call void @__enzyme_gsl_fft_scale_hc_f64_i64(ptr {{.+}}, i64 %stride, i64 %n, double 5.000000e-01)
; CHECK:   call i32 @gsl_fft_halfcomplex_backward(ptr {{.+}}, i64 %stride, i64 %n, ptr %[[VHCWT]], ptr %[[VWK2]])
; CHECK:   call void @gsl_fft_halfcomplex_wavetable_free(ptr %[[VHCWT]])
; CHECK:   call void @gsl_fft_real_workspace_free(ptr %[[VWK2]])
; CHECK:   ret void

; CHECK-LABEL: define internal void @diffehc_inv_fn(ptr %data, ptr %"data'", i64 %stride, i64 %n)
; CHECK:   %[[RWT:.+]] = call ptr @gsl_fft_real_wavetable_alloc(i64 %n)
; CHECK:   %[[RWK:.+]] = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
; CHECK:   %{{.+}} = call i32 @gsl_fft_real_transform(ptr %"data'", i64 %stride, i64 %n, ptr %[[RWT]], ptr %[[RWK]])
; CHECK:   call void @__enzyme_gsl_fft_scale_hc_f64_i64(ptr %"data'", i64 %stride, i64 %n, double 2.000000e+00)
; CHECK:   call void @__enzyme_gsl_fft_scale_real_f64_i64(ptr %"data'", i64 %stride, i64 %n, double %{{.+}})
; CHECK:   call void @gsl_fft_real_wavetable_free(ptr %[[RWT]])
; CHECK:   call void @gsl_fft_real_workspace_free(ptr %[[RWK]])
; CHECK:   ret void

; CHECK-LABEL: define internal void @differadix2_fn(ptr %data, ptr %"data'", ptr %cdata, ptr %"cdata'", i64 %stride, i64 %n)
; CHECK:   call i32 @gsl_fft_real_radix2_transform(ptr %"data'", i64 %stride, i64 %n)
; CHECK:   call void @__enzyme_gsl_fft_scale_hc_radix2_f64_i64(ptr %"data'", i64 %stride, i64 %n, double 2.000000e+00)
; CHECK:   call void @__enzyme_gsl_fft_scale_real_f64_i64(ptr %"data'", i64 %stride, i64 %n, double %{{.+}})
; CHECK:   call void @__enzyme_gsl_fft_hc_radix2_unpack_rev_f64_i64(ptr %"data'", ptr %"cdata'", i64 %stride, i64 %n)
; CHECK:   call void @__enzyme_gsl_fft_scale_hc_radix2_f64_i64(ptr %"data'", i64 %stride, i64 %n, double 5.000000e-01)
; CHECK:   call i32 @gsl_fft_halfcomplex_radix2_backward(ptr %"data'", i64 %stride, i64 %n)
; CHECK:   ret void

; CHECK-LABEL: define internal void @diffecomplex_fn(ptr %data, ptr %"data'", i64 %stride, i64 %n, i32 %sign)
; CHECK:   %[[NEGSIGN:.+]] = sub i32 0, %sign
; CHECK:   call i32 @gsl_fft_complex_transform(ptr %"data'", i64 %stride, i64 %n, ptr %{{.+}}, ptr %{{.+}}, i32 %[[NEGSIGN]])
; CHECK:   call i32 @gsl_fft_complex_forward(ptr %"data'", i64 %stride, i64 %n, ptr %{{.+}}, ptr %{{.+}})
; CHECK:   call void @__enzyme_gsl_fft_scale_complex_f64_i64(ptr %"data'", i64 %stride, i64 %n, double %{{.+}})
; CHECK:   call i32 @gsl_fft_complex_backward(ptr %"data'", i64 %stride, i64 %n, ptr %{{.+}}, ptr %{{.+}})
; CHECK:   ret void

; CHECK-LABEL: define internal void @diffedemo_pipeline(ptr %data, ptr %"data'", ptr %cdata, ptr %"cdata'", i64 %n, double %differeturn)
; CHECK:   call fast double @__enzyme_gsl_stats_var_rev_f64_i64(ptr %cdata, i64 2, ptr %"cdata'", i64 2, i64 %n, double %m, double %{{.+}})
; CHECK:   call void @__enzyme_gsl_stats_mean_rev_f64_i64(ptr %"cdata'", i64 2, i64 %n, double %{{.+}})
; CHECK:   call void @__enzyme_gsl_fft_hc_unpack_rev_f64_i64(ptr %"data'", ptr %"cdata'", i64 1, i64 %n)
; CHECK:   call void @__enzyme_gsl_fft_scale_hc_f64_i64(ptr %"data'", i64 1, i64 %n, double 5.000000e-01)
; CHECK:   call i32 @gsl_fft_halfcomplex_backward(ptr %"data'", i64 1, i64 %n, ptr %{{.+}}, ptr %{{.+}})
; CHECK:   ret void

; CHECK-LABEL: define internal void @diffeunpack_const_src(ptr %src, ptr %dst, ptr %"dst'", i64 %stride, i64 %n)
; CHECK:   call void @__enzyme_gsl_fft_zero_complex_f64_i64(ptr %"dst'", i64 %stride, i64 %n)
; CHECK:   ret void
