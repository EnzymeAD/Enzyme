; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg),verify" -S | FileCheck %s; fi

declare ptr @gsl_fft_real_wavetable_alloc(i64)
declare void @gsl_fft_real_wavetable_free(ptr)
declare ptr @gsl_fft_real_workspace_alloc(i64)
declare void @gsl_fft_real_workspace_free(ptr)
declare ptr @gsl_fft_complex_wavetable_alloc(i64)
declare void @gsl_fft_complex_wavetable_free(ptr)
declare ptr @gsl_fft_complex_workspace_alloc(i64)
declare void @gsl_fft_complex_workspace_free(ptr)

declare i32 @gsl_fft_real_transform(ptr, i64, i64, ptr, ptr)
declare i32 @gsl_fft_real_radix2_transform(ptr, i64, i64)
declare i32 @gsl_fft_complex_transform(ptr, i64, i64, ptr, ptr, i32)
declare i32 @gsl_fft_halfcomplex_unpack(ptr, ptr, i64, i64)

declare void @__enzyme_fwddiff(...)

define void @real_and_unpack(ptr %data, ptr %cdata, i64 %stride, i64 %n) {
entry:
  %wt = call ptr @gsl_fft_real_wavetable_alloc(i64 %n)
  %wk = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
  %r0 = call i32 @gsl_fft_real_transform(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk)
  %r1 = call i32 @gsl_fft_halfcomplex_unpack(ptr %data, ptr %cdata, i64 %stride, i64 %n)
  call void @gsl_fft_real_wavetable_free(ptr %wt)
  call void @gsl_fft_real_workspace_free(ptr %wk)
  ret void
}

define void @const_src_unpack(ptr %hc, ptr %c, i64 %stride, i64 %n) {
entry:
  %r = call i32 @gsl_fft_halfcomplex_unpack(ptr %hc, ptr %c, i64 %stride, i64 %n)
  ret void
}

define void @complex_and_radix2(ptr %cdata, ptr %rdata, i64 %stride, i64 %n, i32 %sign) {
entry:
  %wt = call ptr @gsl_fft_complex_wavetable_alloc(i64 %n)
  %wk = call ptr @gsl_fft_complex_workspace_alloc(i64 %n)
  %r0 = call i32 @gsl_fft_complex_transform(ptr %cdata, i64 %stride, i64 %n, ptr %wt, ptr %wk, i32 %sign)
  call void @gsl_fft_complex_wavetable_free(ptr %wt)
  call void @gsl_fft_complex_workspace_free(ptr %wk)
  %r1 = call i32 @gsl_fft_real_radix2_transform(ptr %rdata, i64 %stride, i64 %n)
  ret void
}

define void @caller(ptr %data, ptr %ddata, ptr %ddata2, ptr %cdata, ptr %dcdata, ptr %dcdata2, i64 %stride, i64 %n, i32 %sign) {
entry:
  call void (...) @__enzyme_fwddiff(ptr @real_and_unpack, metadata !"enzyme_dup", ptr %data, ptr %ddata, metadata !"enzyme_dup", ptr %cdata, ptr %dcdata, i64 %stride, i64 %n)
  call void (...) @__enzyme_fwddiff(ptr @real_and_unpack, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %data, ptr %ddata, ptr %ddata2, metadata !"enzyme_dup", ptr %cdata, ptr %dcdata, ptr %dcdata2, i64 %stride, i64 %n)
  call void (...) @__enzyme_fwddiff(ptr @const_src_unpack, metadata !"enzyme_const", ptr %data, metadata !"enzyme_dup", ptr %cdata, ptr %dcdata, i64 %stride, i64 %n)
  call void (...) @__enzyme_fwddiff(ptr @complex_and_radix2, metadata !"enzyme_dup", ptr %cdata, ptr %dcdata, metadata !"enzyme_dup", ptr %data, ptr %ddata, i64 %stride, i64 %n, i32 %sign)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffereal_and_unpack(ptr %data, ptr %"data'", ptr %cdata, ptr %"cdata'", i64 %stride, i64 %n)
; CHECK:   %wt = call ptr @gsl_fft_real_wavetable_alloc(i64 %n)
; CHECK:   %wk = call ptr @gsl_fft_real_workspace_alloc(i64 %n)
; CHECK:   %r0 = call i32 @gsl_fft_real_transform(ptr %data, i64 %stride, i64 %n, ptr %wt, ptr %wk)
; CHECK:   call i32 @gsl_fft_real_transform(ptr %"data'", i64 %stride, i64 %n, ptr %wt, ptr %wk)
; CHECK:   %r1 = call i32 @gsl_fft_halfcomplex_unpack(ptr %data, ptr %cdata, i64 %stride, i64 %n)
; CHECK:   call i32 @gsl_fft_halfcomplex_unpack(ptr %"data'", ptr %"cdata'", i64 %stride, i64 %n)
; CHECK:   call void @gsl_fft_real_wavetable_free(ptr %wt)
; CHECK:   call void @gsl_fft_real_workspace_free(ptr %wk)
; CHECK:   ret void

; CHECK-LABEL: define internal void @fwddiffe2real_and_unpack(ptr %data, [2 x ptr] %"data'", ptr %cdata, [2 x ptr] %"cdata'", i64 %stride, i64 %n)
; CHECK:   call i32 @gsl_fft_real_transform(ptr %{{.+}}, i64 %stride, i64 %n, ptr %wt, ptr %wk)
; CHECK:   call i32 @gsl_fft_real_transform(ptr %{{.+}}, i64 %stride, i64 %n, ptr %wt, ptr %wk)
; CHECK:   call i32 @gsl_fft_halfcomplex_unpack(ptr %{{.+}}, ptr %{{.+}}, i64 %stride, i64 %n)
; CHECK:   call i32 @gsl_fft_halfcomplex_unpack(ptr %{{.+}}, ptr %{{.+}}, i64 %stride, i64 %n)
; CHECK:   ret void

; CHECK-LABEL: define internal void @fwddiffeconst_src_unpack(ptr %hc, ptr %c, ptr %"c'", i64 %stride, i64 %n)
; CHECK:   call i32 @gsl_fft_halfcomplex_unpack(ptr %hc, ptr %c, i64 %stride, i64 %n)
; CHECK:   call void @__enzyme_gsl_fft_zero_complex_f64_i64(ptr %"c'", i64 %stride, i64 %n)
; CHECK:   ret void

; CHECK-LABEL: define internal void @fwddiffecomplex_and_radix2(ptr %cdata, ptr %"cdata'", ptr %rdata, ptr %"rdata'", i64 %stride, i64 %n, i32 %sign)
; CHECK:   call i32 @gsl_fft_complex_transform(ptr %cdata, i64 %stride, i64 %n, ptr %wt, ptr %wk, i32 %sign)
; CHECK:   call i32 @gsl_fft_complex_transform(ptr %"cdata'", i64 %stride, i64 %n, ptr %wt, ptr %wk, i32 %sign)
; CHECK:   call i32 @gsl_fft_real_radix2_transform(ptr %rdata, i64 %stride, i64 %n)
; CHECK:   call i32 @gsl_fft_real_radix2_transform(ptr %"rdata'", i64 %stride, i64 %n)
; CHECK:   ret void
