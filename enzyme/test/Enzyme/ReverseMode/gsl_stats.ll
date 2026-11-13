; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg),verify" -S | FileCheck %s; fi

declare double @gsl_stats_mean(ptr, i64, i64)
declare double @gsl_stats_variance(ptr, i64, i64)
declare double @gsl_stats_variance_m(ptr, i64, i64, double)
declare double @gsl_stats_variance_with_fixed_mean(ptr, i64, i64, double)
declare double @gsl_stats_sd(ptr, i64, i64)
declare double @gsl_stats_sd_m(ptr, i64, i64, double)
declare double @gsl_stats_sd_with_fixed_mean(ptr, i64, i64, double)
declare double @gsl_stats_tss(ptr, i64, i64)
declare double @gsl_stats_tss_m(ptr, i64, i64, double)
declare double @gsl_stats_max(ptr, i64, i64)
declare double @gsl_stats_min(ptr, i64, i64)
declare i64 @gsl_stats_max_index(ptr, i64, i64)
declare double @gsl_stats_float_mean(ptr, i64, i64)
declare double @gsl_stats_float_sd_m(ptr, i64, i64, double)

declare double @__enzyme_autodiff(...)

define internal double @sd_overwrite_callee(ptr %data, i64 %stride, i64 %n, double %m) noinline {
entry:
  %sd = call double @gsl_stats_sd_m(ptr %data, i64 %stride, i64 %n, double %m)
  store double 0.000000e+00, ptr %data, align 8
  ret double %sd
}

define double @stats_all(ptr %data, i64 %stride, i64 %n, double %m) {
entry:
  %v0 = call double @gsl_stats_mean(ptr %data, i64 %stride, i64 %n)
  %v1 = call double @gsl_stats_variance(ptr %data, i64 %stride, i64 %n)
  %v2 = call double @gsl_stats_variance_m(ptr %data, i64 %stride, i64 %n, double %m)
  %v3 = call double @gsl_stats_variance_with_fixed_mean(ptr %data, i64 %stride, i64 %n, double %m)
  %v4 = call double @gsl_stats_sd(ptr %data, i64 %stride, i64 %n)
  %v5 = call double @gsl_stats_sd_with_fixed_mean(ptr %data, i64 %stride, i64 %n, double %m)
  %v6 = call double @gsl_stats_tss(ptr %data, i64 %stride, i64 %n)
  %v7 = call double @gsl_stats_tss_m(ptr %data, i64 %stride, i64 %n, double %m)
  %v8 = call double @gsl_stats_max(ptr %data, i64 %stride, i64 %n)
  %v9 = call double @gsl_stats_min(ptr %data, i64 %stride, i64 %n)
  %v10 = call double @sd_overwrite_callee(ptr %data, i64 %stride, i64 %n, double %m)
  %v11 = call double @gsl_stats_mean(ptr %data, i64 %stride, i64 %n)
  %s0 = fadd double %v0, %v1
  %s1 = fadd double %s0, %v2
  %s2 = fadd double %s1, %v3
  %s3 = fadd double %s2, %v4
  %s4 = fadd double %s3, %v5
  %s5 = fadd double %s4, %v6
  %s6 = fadd double %s5, %v7
  %s7 = fadd double %s6, %v8
  %s8 = fadd double %s7, %v9
  %s9 = fadd double %s8, %v10
  %s10 = fadd double %s9, %v11
  ret double %s10
}

define double @stats_float(ptr %data, i64 %stride, i64 %n, double %m) {
entry:
  %v0 = call double @gsl_stats_float_mean(ptr %data, i64 %stride, i64 %n)
  %v1 = call double @gsl_stats_float_sd_m(ptr %data, i64 %stride, i64 %n, double %m)
  %s = fadd double %v0, %v1
  ret double %s
}

define double @stats_const_data_active_mean(ptr %data, i64 %stride, i64 %n, double %m) {
entry:
  %v = call double @gsl_stats_variance_m(ptr %data, i64 %stride, i64 %n, double %m)
  ret double %v
}

define double @caller(ptr %data, ptr %ddata, ptr %fdata, ptr %dfdata, i64 %stride, i64 %n, double %m) {
entry:
  %r0 = call double (...) @__enzyme_autodiff(ptr @stats_all, metadata !"enzyme_dup", ptr %data, ptr %ddata, i64 %stride, i64 %n, double %m)
  %r1 = call double (...) @__enzyme_autodiff(ptr @stats_float, metadata !"enzyme_dup", ptr %fdata, ptr %dfdata, i64 %stride, i64 %n, double %m)
  %r2 = call double (...) @__enzyme_autodiff(ptr @stats_const_data_active_mean, metadata !"enzyme_const", ptr %data, i64 %stride, i64 %n, double %m)
  %s = fadd double %r0, %r1
  %s2 = fadd double %s, %r2
  ret double %s2
}

; CHECK-LABEL: define internal { double } @diffestats_all(
; CHECK:   call { ptr, double } @augmented_sd_overwrite_callee(ptr %data, ptr %"data'", i64 %stride, i64 %n, double %m)
; CHECK:   call void @__enzyme_gsl_stats_mean_rev_f64_i64(ptr %"data'", i64 %stride, i64 %n, double %differeturn)
; CHECK:   call { double } @diffesd_overwrite_callee(ptr %data, ptr %"data'", i64 %stride, i64 %n, double %m, double %differeturn, { ptr, double } %{{.+}})
; CHECK:   call void @__enzyme_gsl_stats_minmax_rev_f64_i64(ptr %"data'", i64 %stride, i64 %n, i64 %{{.+}}, double %differeturn)
; CHECK:   call void @__enzyme_gsl_stats_minmax_rev_f64_i64(ptr %"data'", i64 %stride, i64 %n, i64 %{{.+}}, double %differeturn)
; CHECK:   call fast double @__enzyme_gsl_stats_var_rev_f64_i64(ptr %{{.+}}, i64 1, ptr %"data'", i64 %stride, i64 %n, double %m, double %{{.+}})
; CHECK:   call void @__enzyme_gsl_stats_mean_rev_f64_i64(ptr %"data'", i64 %stride, i64 %n, double %differeturn)
; CHECK:   ret { double }

; CHECK-LABEL: define internal { ptr, double } @augmented_sd_overwrite_callee(
; CHECK:   %[[COPY:.+]] = tail call noalias nonnull ptr @malloc(i64 %{{.+}})
; CHECK:   call void @__enzyme_gsl_copy_strided_f64_i64(ptr %[[COPY]], ptr %data, i64 %stride, i64 %n)
; CHECK:   %sd = call double @gsl_stats_sd_m(ptr %data, i64 %stride, i64 %n, double %m)
; CHECK:   store double 0.000000e+00, ptr %data, align 8
; CHECK:   ret { ptr, double }

; CHECK-LABEL: define internal { double } @diffesd_overwrite_callee(
; CHECK:   %[[COPYR:.+]] = extractvalue { ptr, double } %tapeArg, 0
; CHECK:   %[[SDR:.+]] = extractvalue { ptr, double } %tapeArg, 1
; CHECK:   %[[DM:.+]] = call fast double @__enzyme_gsl_stats_var_rev_f64_i64(ptr %[[COPYR]], i64 1, ptr %"data'", i64 %stride, i64 %n, double %m, double %{{.+}})
; CHECK:   tail call void @free(ptr nonnull %[[COPYR]])
; CHECK:   %[[RET:.+]] = insertvalue { double } {{(undef|poison)}}, double %[[DM]], 0
; CHECK:   ret { double } %[[RET]]

; CHECK-LABEL: define internal { double } @diffestats_float(
; CHECK:   call fast double @__enzyme_gsl_stats_var_rev_f32_i64(ptr %data, i64 %stride, ptr %"data'", i64 %stride, i64 %n, double %m, double %{{.+}})
; CHECK:   call void @__enzyme_gsl_stats_mean_rev_f32_i64(ptr %"data'", i64 %stride, i64 %n, double %differeturn)
; CHECK:   ret { double }

; CHECK-LABEL: define internal { double } @diffestats_const_data_active_mean(
; CHECK:   %[[DM2:.+]] = call fast double @__enzyme_gsl_stats_var_rev_f64_i64(ptr %data, i64 %stride, ptr null, i64 %stride, i64 %n, double %m, double %{{.+}})
; CHECK:   %[[RET2:.+]] = insertvalue { double } {{(undef|poison)}}, double %[[DM2]], 0
; CHECK:   ret { double } %[[RET2]]
