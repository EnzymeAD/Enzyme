; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg),verify" -S | FileCheck %s; fi

declare double @gsl_stats_mean(ptr, i64, i64)
declare double @gsl_stats_variance(ptr, i64, i64)
declare double @gsl_stats_sd_m(ptr, i64, i64, double)
declare double @gsl_stats_tss_m(ptr, i64, i64, double)
declare double @gsl_stats_max(ptr, i64, i64)
declare double @gsl_stats_float_sd_m(ptr, i64, i64, double)

declare double @__enzyme_fwddiff(...)

define double @stats_fwd(ptr %data, i64 %stride, i64 %n, double %m) {
entry:
  %v0 = call double @gsl_stats_mean(ptr %data, i64 %stride, i64 %n)
  %v1 = call double @gsl_stats_variance(ptr %data, i64 %stride, i64 %n)
  %v2 = call double @gsl_stats_sd_m(ptr %data, i64 %stride, i64 %n, double %m)
  %v3 = call double @gsl_stats_tss_m(ptr %data, i64 %stride, i64 %n, double %m)
  %v4 = call double @gsl_stats_max(ptr %data, i64 %stride, i64 %n)
  %s0 = fadd double %v0, %v1
  %s1 = fadd double %s0, %v2
  %s2 = fadd double %s1, %v3
  %s3 = fadd double %s2, %v4
  ret double %s3
}

define double @stats_float_fwd(ptr %fdata, i64 %stride, i64 %n, double %m) {
entry:
  %v = call double @gsl_stats_float_sd_m(ptr %fdata, i64 %stride, i64 %n, double %m)
  ret double %v
}

define void @caller(ptr %data, ptr %ddata, ptr %ddata2, ptr %fdata, ptr %dfdata, i64 %stride, i64 %n, double %m, double %dm, double %dm2) {
entry:
  %r0 = call double (...) @__enzyme_fwddiff(ptr @stats_fwd, metadata !"enzyme_dup", ptr %data, ptr %ddata, i64 %stride, i64 %n, metadata !"enzyme_dup", double %m, double %dm)
  %r1 = call [2 x double] (...) @__enzyme_fwddiff(ptr @stats_fwd, metadata !"enzyme_width", i64 2, metadata !"enzyme_dup", ptr %data, ptr %ddata, ptr %ddata2, i64 %stride, i64 %n, metadata !"enzyme_dup", double %m, double %dm, double %dm2)
  %r2 = call double (...) @__enzyme_fwddiff(ptr @stats_float_fwd, metadata !"enzyme_dup", ptr %fdata, ptr %dfdata, i64 %stride, i64 %n, metadata !"enzyme_dup", double %m, double %dm)
  ret void
}

; CHECK-LABEL: define internal double @fwddiffestats_fwd(
; CHECK:   call fast double @gsl_stats_mean(ptr %"data'", i64 %stride, i64 %n)
; CHECK:   call fast double @__enzyme_gsl_stats_var_fwd_f64_i64(ptr %data, i64 %stride, ptr %"data'", i64 %stride, i64 %n, double %{{.+}}, double 0.000000e+00)
; CHECK:   call fast double @__enzyme_gsl_stats_var_fwd_f64_i64(ptr %data, i64 %stride, ptr %"data'", i64 %stride, i64 %n, double %m, double %"m'")
; CHECK:   call fast double @__enzyme_gsl_stats_var_fwd_f64_i64(ptr %data, i64 %stride, ptr %"data'", i64 %stride, i64 %n, double %m, double %"m'")
; CHECK:   %[[MAXI:.+]] = call i64 @gsl_stats_max_index(ptr %data, i64 %stride, i64 %n)
; CHECK:   call fast double @__enzyme_gsl_stats_minmax_fwd_f64_i64(ptr %"data'", i64 %stride, i64 %n, i64 %[[MAXI]])
; CHECK:   ret double

; CHECK-LABEL: define internal [2 x double] @fwddiffe2stats_fwd(
; CHECK:   call fast double @gsl_stats_mean(ptr %{{.+}}, i64 %stride, i64 %n)
; CHECK:   call fast double @gsl_stats_mean(ptr %{{.+}}, i64 %stride, i64 %n)
; CHECK:   ret [2 x double]

; CHECK-LABEL: define internal double @fwddiffestats_float_fwd(
; CHECK:   %v = call double @gsl_stats_float_sd_m(ptr %fdata, i64 %stride, i64 %n, double %m)
; CHECK:   call fast double @__enzyme_gsl_stats_var_fwd_f32_i64(ptr %fdata, i64 %stride, ptr %"fdata'", i64 %stride, i64 %n, double %m, double %"m'")
; CHECK:   ret double
