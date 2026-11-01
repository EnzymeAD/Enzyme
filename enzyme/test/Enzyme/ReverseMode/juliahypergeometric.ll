; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Reduced from Enzyme.jl: the reverse gradient of HypergeometricFunctions'
; _2F1 with respect to all four arguments (EnzymeAD/Enzyme.jl#3298). This is
; the z -> infinity continuation, BInf, whose derivative Enzyme spends almost
; all of its time on: lookupM tries to recompute values that live inside the
; loops here, rebuilding phis into fresh blocks whose siblings do not share
; lookup results, so the exploration multiplies out.
;
; A compile time test. With the recompute bound this differentiates in about
; a second; without it, this reduced module takes ~50x longer, and the
; unreduced one does not finish at all.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

; Function Attrs: nofree
declare void @julia.safepoint(ptr) local_unnamed_addr #0
declare nonnull ptr @julia.get_pgcstack() local_unnamed_addr "enzyme_inactive"

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #1

define dso_local double @julia_gamma_3408(double %"_x::Float64") local_unnamed_addr {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %current_task = getelementptr inbounds i8, ptr %pgcstack, i64 -152
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(ptr %safepoint)
  fence syncscope("singlethread") seq_cst
  %1 = fcmp olt double %"_x::Float64", 0.000000e+00
  %2 = fsub double %"_x::Float64", %"_x::Float64"
  %3 = fcmp uno double %2, 0.000000e+00
  %or.cond.not = or i1 %1, %3
  ret double 0.000000e+00
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.rint.f64(double) #1

define hidden fastcc double @julia_G_3448(double %"z::Float64", double %"\CF\B5::Float64") unnamed_addr {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %current_task = getelementptr inbounds i8, ptr %pgcstack, i64 -152
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(ptr %safepoint)
  fence syncscope("singlethread") seq_cst
  %1 = call double @llvm.rint.f64(double %"z::Float64")
  %2 = fcmp ult double %1, 0xC3E0000000000000
  %3 = fcmp uge double %1, 0x43E0000000000000
  %narrow.not = or i1 %2, %3
  %4 = fsub double %1, %1
  %5 = fcmp une double %4, 0.000000e+00
  %or.cond = or i1 %narrow.not, %5
  ret double 0.000000e+00
}

declare dso_local double @julia___3420(double, i64) local_unnamed_addr #2

define hidden fastcc double @julia_P_3542(double %"z::Float64", double %"\CF\B5::Float64", i64 %"m::Int64") unnamed_addr {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(ptr %safepoint)
  fence syncscope("singlethread") seq_cst
  %1 = call double @llvm.rint.f64(double %"z::Float64")
  %2 = fcmp ult double %1, 0xC3E0000000000000
  %3 = fcmp uge double %1, 0x43E0000000000000
  %narrow.not = or i1 %2, %3
  %4 = fsub double %1, %1
  %5 = fcmp une double %4, 0.000000e+00
  %or.cond = or i1 %narrow.not, %5
  %6 = fptosi double %1 to i64
  %7 = freeze i64 %6
  %8 = sub i64 0, %7
  %9 = fcmp une double %"\CF\B5::Float64", 0.000000e+00
  %10 = icmp slt i64 %8, 0
  %11 = icmp sge i64 %8, %"m::Int64"
  %narrow.not93 = select i1 %10, i1 true, i1 %11
  %.not126 = icmp sgt i64 %"m::Int64", 0
  br i1 %.not126, label %L40.lr.ph, label %L56

L40.lr.ph:                                        ; preds = %top
  %xtraiter = and i64 %"m::Int64", 7
  %12 = icmp ult i64 %"m::Int64", 8
  br i1 %12, label %L35.L56_crit_edge.unr-lcssa, label %L40.lr.ph.new

L40.lr.ph.new:                                    ; preds = %L40.lr.ph
  %13 = add nsw i64 %"m::Int64", -8
  %14 = lshr i64 %13, 3
  %15 = add nuw nsw i64 %14, 1
  %xtraiter221 = and i64 %15, 3
  %16 = icmp ult i64 %13, 24
  br i1 %16, label %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa, label %L40.lr.ph.new.new

L40.lr.ph.new.new:                                ; preds = %L40.lr.ph.new
  %unroll_iter239 = and i64 %15, 4611686018427387900
  br label %L40

L40:                                              ; preds = %L52.6.3, %L40.lr.ph.new.new
  %value_phi4129 = phi i64 [ 0, %L40.lr.ph.new.new ], [ %value_phi7.7.3, %L52.6.3 ]
  %value_phi3128 = phi double [ 1.000000e+00, %L40.lr.ph.new.new ], [ %value_phi6.6.3, %L52.6.3 ]
  %value_phi2127 = phi double [ 0.000000e+00, %L40.lr.ph.new.new ], [ %value_phi5.6.3, %L52.6.3 ]
  %niter = phi i64 [ 0, %L40.lr.ph.new.new ], [ %niter.next.3, %L52.6.3 ]
  %.not94 = icmp eq i64 %value_phi4129, %8
  br i1 %.not94, label %L52, label %L44

L44:                                              ; preds = %L40
  %17 = sitofp i64 %value_phi4129 to double
  %18 = fadd double %17, %"z::Float64"
  %19 = fmul double %value_phi3128, %18
  %20 = fdiv double 1.000000e+00, %18
  %21 = fadd double %value_phi2127, %20
  br label %L52

L52:                                              ; preds = %L44, %L40
  %value_phi5 = phi double [ %21, %L44 ], [ %value_phi2127, %L40 ]
  %value_phi6 = phi double [ %19, %L44 ], [ %value_phi3128, %L40 ]
  %value_phi7 = or disjoint i64 %value_phi4129, 1
  %.not94.1 = icmp eq i64 %value_phi7, %8
  br i1 %.not94.1, label %L52.1, label %L44.1

L44.1:                                            ; preds = %L52
  %22 = sitofp i64 %value_phi7 to double
  %23 = fadd double %22, %"z::Float64"
  %24 = fmul double %23, %value_phi6
  %25 = fdiv double 1.000000e+00, %23
  %26 = fadd double %25, %value_phi5
  br label %L52.1

L52.1:                                            ; preds = %L44.1, %L52
  %value_phi5.1 = phi double [ %26, %L44.1 ], [ %value_phi5, %L52 ]
  %value_phi6.1 = phi double [ %24, %L44.1 ], [ %value_phi6, %L52 ]
  %value_phi7.1 = or disjoint i64 %value_phi4129, 2
  %.not94.2 = icmp eq i64 %value_phi7.1, %8
  br i1 %.not94.2, label %L52.2, label %L44.2

L44.2:                                            ; preds = %L52.1
  %27 = sitofp i64 %value_phi7.1 to double
  %28 = fadd double %27, %"z::Float64"
  %29 = fmul double %28, %value_phi6.1
  %30 = fdiv double 1.000000e+00, %28
  %31 = fadd double %30, %value_phi5.1
  br label %L52.2

L52.2:                                            ; preds = %L44.2, %L52.1
  %value_phi5.2 = phi double [ %31, %L44.2 ], [ %value_phi5.1, %L52.1 ]
  %value_phi6.2 = phi double [ %29, %L44.2 ], [ %value_phi6.1, %L52.1 ]
  %value_phi7.2 = or disjoint i64 %value_phi4129, 3
  %.not94.3 = icmp eq i64 %value_phi7.2, %8
  %value_phi7.3 = or disjoint i64 %value_phi4129, 4
  %.not94.4 = icmp eq i64 %value_phi7.3, %8
  br i1 %.not94.4, label %L52.4, label %L44.4

L44.4:                                            ; preds = %L52.2
  %32 = sitofp i64 %value_phi7.3 to double
  %33 = fadd double %32, %"z::Float64"
  %34 = fmul double %33, %value_phi6.2
  %35 = fdiv double 1.000000e+00, %33
  %36 = fadd double %35, %value_phi5.2
  br label %L52.4

L52.4:                                            ; preds = %L44.4, %L52.2
  %value_phi5.4 = phi double [ %36, %L44.4 ], [ %value_phi5.2, %L52.2 ]
  %value_phi6.4 = phi double [ %34, %L44.4 ], [ %value_phi6.2, %L52.2 ]
  %value_phi7.4 = or disjoint i64 %value_phi4129, 5
  %.not94.5 = icmp eq i64 %value_phi7.4, %8
  br i1 %.not94.5, label %L52.5, label %L44.5

L44.5:                                            ; preds = %L52.4
  %37 = sitofp i64 %value_phi7.4 to double
  %38 = fadd double %37, %"z::Float64"
  %39 = fmul double %38, %value_phi6.4
  %40 = fdiv double 1.000000e+00, %38
  %41 = fadd double %40, %value_phi5.4
  br label %L52.5

L52.5:                                            ; preds = %L44.5, %L52.4
  %value_phi5.5 = phi double [ %41, %L44.5 ], [ %value_phi5.4, %L52.4 ]
  %value_phi6.5 = phi double [ %39, %L44.5 ], [ %value_phi6.4, %L52.4 ]
  %value_phi7.5 = or disjoint i64 %value_phi4129, 6
  %.not94.6 = icmp eq i64 %value_phi7.5, %8
  %value_phi7.6 = or disjoint i64 %value_phi4129, 7
  %.not94.7 = icmp eq i64 %value_phi7.6, %8
  br i1 %.not94.7, label %L52.7, label %L44.7

L44.7:                                            ; preds = %L52.5
  %42 = sitofp i64 %value_phi7.6 to double
  %43 = fadd double %42, %"z::Float64"
  %44 = fmul double %43, %value_phi6.5
  %45 = fdiv double 1.000000e+00, %43
  %46 = fadd double %45, %value_phi5.5
  br label %L52.7

L52.7:                                            ; preds = %L44.7, %L52.5
  %value_phi5.7 = phi double [ %46, %L44.7 ], [ %value_phi5.5, %L52.5 ]
  %value_phi6.7 = phi double [ %44, %L44.7 ], [ %value_phi6.5, %L52.5 ]
  %value_phi7.7 = or disjoint i64 %value_phi4129, 8
  %.not94.1240 = icmp eq i64 %value_phi7.7, %8
  br i1 %.not94.1240, label %L52.1245, label %L44.1241

L44.1241:                                         ; preds = %L52.7
  %47 = sitofp i64 %value_phi7.7 to double
  %48 = fadd double %47, %"z::Float64"
  %49 = fmul double %value_phi6.7, %48
  %50 = fdiv double 1.000000e+00, %48
  %51 = fadd double %value_phi5.7, %50
  br label %L52.1245

L52.1245:                                         ; preds = %L44.1241, %L52.7
  %value_phi5.1242 = phi double [ %51, %L44.1241 ], [ %value_phi5.7, %L52.7 ]
  %value_phi6.1243 = phi double [ %49, %L44.1241 ], [ %value_phi6.7, %L52.7 ]
  %value_phi7.1244 = or disjoint i64 %value_phi4129, 9
  %.not94.1.1 = icmp eq i64 %value_phi7.1244, %8
  %value_phi7.1.1 = or disjoint i64 %value_phi4129, 10
  %.not94.2.1 = icmp eq i64 %value_phi7.1.1, %8
  br i1 %.not94.2.1, label %L52.2.1, label %L44.2.1

L44.2.1:                                          ; preds = %L52.1245
  %52 = sitofp i64 %value_phi7.1.1 to double
  %53 = fadd double %52, %"z::Float64"
  %54 = fmul double %53, %value_phi6.1243
  %55 = fdiv double 1.000000e+00, %53
  %56 = fadd double %55, %value_phi5.1242
  br label %L52.2.1

L52.2.1:                                          ; preds = %L44.2.1, %L52.1245
  %value_phi5.2.1 = phi double [ %56, %L44.2.1 ], [ %value_phi5.1242, %L52.1245 ]
  %value_phi6.2.1 = phi double [ %54, %L44.2.1 ], [ %value_phi6.1243, %L52.1245 ]
  %value_phi7.2.1 = or disjoint i64 %value_phi4129, 11
  %.not94.3.1 = icmp eq i64 %value_phi7.2.1, %8
  br i1 %.not94.3.1, label %L52.3.1, label %L44.3.1

L44.3.1:                                          ; preds = %L52.2.1
  %57 = sitofp i64 %value_phi7.2.1 to double
  %58 = fadd double %57, %"z::Float64"
  %59 = fmul double %58, %value_phi6.2.1
  %60 = fdiv double 1.000000e+00, %58
  %61 = fadd double %60, %value_phi5.2.1
  br label %L52.3.1

L52.3.1:                                          ; preds = %L44.3.1, %L52.2.1
  %value_phi5.3.1 = phi double [ %61, %L44.3.1 ], [ %value_phi5.2.1, %L52.2.1 ]
  %value_phi6.3.1 = phi double [ %59, %L44.3.1 ], [ %value_phi6.2.1, %L52.2.1 ]
  %value_phi7.3.1 = or disjoint i64 %value_phi4129, 12
  %.not94.4.1 = icmp eq i64 %value_phi7.3.1, %8
  %value_phi7.4.1 = or disjoint i64 %value_phi4129, 13
  %.not94.5.1 = icmp eq i64 %value_phi7.4.1, %8
  br i1 %.not94.5.1, label %L52.5.1, label %L44.5.1

L44.5.1:                                          ; preds = %L52.3.1
  %62 = sitofp i64 %value_phi7.4.1 to double
  %63 = fadd double %62, %"z::Float64"
  %64 = fmul double %63, %value_phi6.3.1
  %65 = fdiv double 1.000000e+00, %63
  %66 = fadd double %65, %value_phi5.3.1
  br label %L52.5.1

L52.5.1:                                          ; preds = %L44.5.1, %L52.3.1
  %value_phi5.5.1 = phi double [ %66, %L44.5.1 ], [ %value_phi5.3.1, %L52.3.1 ]
  %value_phi6.5.1 = phi double [ %64, %L44.5.1 ], [ %value_phi6.3.1, %L52.3.1 ]
  %value_phi7.5.1 = or disjoint i64 %value_phi4129, 14
  %.not94.6.1 = icmp eq i64 %value_phi7.5.1, %8
  %value_phi7.6.1 = or disjoint i64 %value_phi4129, 15
  %.not94.7.1 = icmp eq i64 %value_phi7.6.1, %8
  br i1 %.not94.7.1, label %L52.7.1, label %L44.7.1

L44.7.1:                                          ; preds = %L52.5.1
  %67 = sitofp i64 %value_phi7.6.1 to double
  %68 = fadd double %67, %"z::Float64"
  %69 = fmul double %68, %value_phi6.5.1
  %70 = fdiv double 1.000000e+00, %68
  %71 = fadd double %70, %value_phi5.5.1
  br label %L52.7.1

L52.7.1:                                          ; preds = %L44.7.1, %L52.5.1
  %value_phi5.7.1 = phi double [ %71, %L44.7.1 ], [ %value_phi5.5.1, %L52.5.1 ]
  %value_phi6.7.1 = phi double [ %69, %L44.7.1 ], [ %value_phi6.5.1, %L52.5.1 ]
  %value_phi7.7.1 = or disjoint i64 %value_phi4129, 16
  %.not94.2246 = icmp eq i64 %value_phi7.7.1, %8
  %value_phi7.2250 = or disjoint i64 %value_phi4129, 17
  %.not94.1.2 = icmp eq i64 %value_phi7.2250, %8
  br i1 %.not94.1.2, label %L52.1.2, label %L44.1.2

L44.1.2:                                          ; preds = %L52.7.1
  %72 = sitofp i64 %value_phi7.2250 to double
  %73 = fadd double %72, %"z::Float64"
  %74 = fmul double %73, %value_phi6.7.1
  %75 = fdiv double 1.000000e+00, %73
  %76 = fadd double %75, %value_phi5.7.1
  br label %L52.1.2

L52.1.2:                                          ; preds = %L44.1.2, %L52.7.1
  %value_phi5.1.2 = phi double [ %76, %L44.1.2 ], [ %value_phi5.7.1, %L52.7.1 ]
  %value_phi6.1.2 = phi double [ %74, %L44.1.2 ], [ %value_phi6.7.1, %L52.7.1 ]
  %value_phi7.1.2 = or disjoint i64 %value_phi4129, 18
  %.not94.2.2 = icmp eq i64 %value_phi7.1.2, %8
  br i1 %.not94.2.2, label %L52.2.2, label %L44.2.2

L44.2.2:                                          ; preds = %L52.1.2
  %77 = sitofp i64 %value_phi7.1.2 to double
  %78 = fadd double %77, %"z::Float64"
  %79 = fmul double %78, %value_phi6.1.2
  %80 = fdiv double 1.000000e+00, %78
  %81 = fadd double %80, %value_phi5.1.2
  br label %L52.2.2

L52.2.2:                                          ; preds = %L44.2.2, %L52.1.2
  %value_phi5.2.2 = phi double [ %81, %L44.2.2 ], [ %value_phi5.1.2, %L52.1.2 ]
  %value_phi6.2.2 = phi double [ %79, %L44.2.2 ], [ %value_phi6.1.2, %L52.1.2 ]
  %value_phi7.2.2 = or disjoint i64 %value_phi4129, 19
  %.not94.3.2 = icmp eq i64 %value_phi7.2.2, %8
  %value_phi7.3.2 = or disjoint i64 %value_phi4129, 20
  %.not94.4.2 = icmp eq i64 %value_phi7.3.2, %8
  br i1 %.not94.4.2, label %L52.4.2, label %L44.4.2

L44.4.2:                                          ; preds = %L52.2.2
  %82 = sitofp i64 %value_phi7.3.2 to double
  %83 = fadd double %82, %"z::Float64"
  %84 = fmul double %83, %value_phi6.2.2
  %85 = fdiv double 1.000000e+00, %83
  %86 = fadd double %85, %value_phi5.2.2
  br label %L52.4.2

L52.4.2:                                          ; preds = %L44.4.2, %L52.2.2
  %value_phi5.4.2 = phi double [ %86, %L44.4.2 ], [ %value_phi5.2.2, %L52.2.2 ]
  %value_phi6.4.2 = phi double [ %84, %L44.4.2 ], [ %value_phi6.2.2, %L52.2.2 ]
  %value_phi7.4.2 = or disjoint i64 %value_phi4129, 21
  %.not94.5.2 = icmp eq i64 %value_phi7.4.2, %8
  br i1 %.not94.5.2, label %L52.5.2, label %L44.5.2

L44.5.2:                                          ; preds = %L52.4.2
  %87 = sitofp i64 %value_phi7.4.2 to double
  %88 = fadd double %87, %"z::Float64"
  %89 = fmul double %88, %value_phi6.4.2
  %90 = fdiv double 1.000000e+00, %88
  %91 = fadd double %90, %value_phi5.4.2
  br label %L52.5.2

L52.5.2:                                          ; preds = %L44.5.2, %L52.4.2
  %value_phi5.5.2 = phi double [ %91, %L44.5.2 ], [ %value_phi5.4.2, %L52.4.2 ]
  %value_phi6.5.2 = phi double [ %89, %L44.5.2 ], [ %value_phi6.4.2, %L52.4.2 ]
  %value_phi7.5.2 = or disjoint i64 %value_phi4129, 22
  %.not94.6.2 = icmp eq i64 %value_phi7.5.2, %8
  %value_phi7.6.2 = or disjoint i64 %value_phi4129, 23
  %.not94.7.2 = icmp eq i64 %value_phi7.6.2, %8
  br i1 %.not94.7.2, label %L52.7.2, label %L44.7.2

L44.7.2:                                          ; preds = %L52.5.2
  %92 = sitofp i64 %value_phi7.6.2 to double
  %93 = fadd double %92, %"z::Float64"
  %94 = fmul double %93, %value_phi6.5.2
  %95 = fdiv double 1.000000e+00, %93
  %96 = fadd double %95, %value_phi5.5.2
  br label %L52.7.2

L52.7.2:                                          ; preds = %L44.7.2, %L52.5.2
  %value_phi5.7.2 = phi double [ %96, %L44.7.2 ], [ %value_phi5.5.2, %L52.5.2 ]
  %value_phi6.7.2 = phi double [ %94, %L44.7.2 ], [ %value_phi6.5.2, %L52.5.2 ]
  %value_phi7.7.2 = or disjoint i64 %value_phi4129, 24
  %.not94.3252 = icmp eq i64 %value_phi7.7.2, %8
  br i1 %.not94.3252, label %L52.3257, label %L44.3253

L44.3253:                                         ; preds = %L52.7.2
  %97 = sitofp i64 %value_phi7.7.2 to double
  %98 = fadd double %97, %"z::Float64"
  %99 = fmul double %value_phi6.7.2, %98
  %100 = fdiv double 1.000000e+00, %98
  %101 = fadd double %value_phi5.7.2, %100
  br label %L52.3257

L52.3257:                                         ; preds = %L44.3253, %L52.7.2
  %value_phi5.3254 = phi double [ %101, %L44.3253 ], [ %value_phi5.7.2, %L52.7.2 ]
  %value_phi6.3255 = phi double [ %99, %L44.3253 ], [ %value_phi6.7.2, %L52.7.2 ]
  %value_phi7.3256 = or disjoint i64 %value_phi4129, 25
  %.not94.1.3 = icmp eq i64 %value_phi7.3256, %8
  %value_phi7.1.3 = or disjoint i64 %value_phi4129, 26
  %.not94.2.3 = icmp eq i64 %value_phi7.1.3, %8
  %value_phi7.2.3 = or disjoint i64 %value_phi4129, 27
  %.not94.3.3 = icmp eq i64 %value_phi7.2.3, %8
  br i1 %.not94.3.3, label %L52.3.3, label %L44.3.3

L44.3.3:                                          ; preds = %L52.3257
  %102 = sitofp i64 %value_phi7.2.3 to double
  %103 = fadd double %102, %"z::Float64"
  %104 = fmul double %103, %value_phi6.3255
  %105 = fdiv double 1.000000e+00, %103
  %106 = fadd double %105, %value_phi5.3254
  br label %L52.3.3

L52.3.3:                                          ; preds = %L44.3.3, %L52.3257
  %value_phi5.3.3 = phi double [ %106, %L44.3.3 ], [ %value_phi5.3254, %L52.3257 ]
  %value_phi6.3.3 = phi double [ %104, %L44.3.3 ], [ %value_phi6.3255, %L52.3257 ]
  %value_phi7.3.3 = or disjoint i64 %value_phi4129, 28
  %.not94.4.3 = icmp eq i64 %value_phi7.3.3, %8
  br i1 %.not94.4.3, label %L52.4.3, label %L44.4.3

L44.4.3:                                          ; preds = %L52.3.3
  %107 = sitofp i64 %value_phi7.3.3 to double
  %108 = fadd double %107, %"z::Float64"
  %109 = fmul double %108, %value_phi6.3.3
  %110 = fdiv double 1.000000e+00, %108
  %111 = fadd double %110, %value_phi5.3.3
  br label %L52.4.3

L52.4.3:                                          ; preds = %L44.4.3, %L52.3.3
  %value_phi5.4.3 = phi double [ %111, %L44.4.3 ], [ %value_phi5.3.3, %L52.3.3 ]
  %value_phi6.4.3 = phi double [ %109, %L44.4.3 ], [ %value_phi6.3.3, %L52.3.3 ]
  %value_phi7.4.3 = or disjoint i64 %value_phi4129, 29
  %.not94.5.3 = icmp eq i64 %value_phi7.4.3, %8
  %value_phi7.5.3 = or disjoint i64 %value_phi4129, 30
  %.not94.6.3 = icmp eq i64 %value_phi7.5.3, %8
  br i1 %.not94.6.3, label %L52.6.3, label %L44.6.3

L44.6.3:                                          ; preds = %L52.4.3
  %112 = sitofp i64 %value_phi7.5.3 to double
  %113 = fadd double %112, %"z::Float64"
  %114 = fmul double %113, %value_phi6.4.3
  %115 = fdiv double 1.000000e+00, %113
  %116 = fadd double %115, %value_phi5.4.3
  br label %L52.6.3

L52.6.3:                                          ; preds = %L44.6.3, %L52.4.3
  %value_phi5.6.3 = phi double [ %116, %L44.6.3 ], [ %value_phi5.4.3, %L52.4.3 ]
  %value_phi6.6.3 = phi double [ %114, %L44.6.3 ], [ %value_phi6.4.3, %L52.4.3 ]
  %value_phi7.6.3 = or disjoint i64 %value_phi4129, 31
  %.not94.7.3 = icmp eq i64 %value_phi7.6.3, %8
  %value_phi7.7.3 = add nuw nsw i64 %value_phi4129, 32
  %niter.next.3 = add i64 %niter, 4
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, %unroll_iter239
  br i1 %niter.ncmp.3, label %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa, label %L40

L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa:   ; preds = %L52.6.3, %L40.lr.ph.new
  %value_phi5.7.lcssa.ph = phi double [ undef, %L40.lr.ph.new ], [ %value_phi5.6.3, %L52.6.3 ]
  %value_phi6.7.lcssa.ph = phi double [ undef, %L40.lr.ph.new ], [ %value_phi6.6.3, %L52.6.3 ]
  %value_phi7.7.lcssa.ph = phi i64 [ undef, %L40.lr.ph.new ], [ %value_phi7.7.3, %L52.6.3 ]
  %value_phi4129.unr233 = phi i64 [ 0, %L40.lr.ph.new ], [ %value_phi7.7.3, %L52.6.3 ]
  %value_phi3128.unr234 = phi double [ 1.000000e+00, %L40.lr.ph.new ], [ %value_phi6.6.3, %L52.6.3 ]
  %value_phi2127.unr235 = phi double [ 0.000000e+00, %L40.lr.ph.new ], [ %value_phi5.6.3, %L52.6.3 ]
  %lcmp.mod.not323 = icmp eq i64 %xtraiter221, 0
  br label %L35.L56_crit_edge.unr-lcssa

L35.L56_crit_edge.unr-lcssa:                      ; preds = %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa, %L40.lr.ph
  %value_phi5.lcssa.ph = phi double [ undef, %L40.lr.ph ], [ %value_phi5.7.lcssa.ph, %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa ]
  %value_phi6.lcssa.ph = phi double [ undef, %L40.lr.ph ], [ %value_phi6.7.lcssa.ph, %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa ]
  %value_phi4129.unr = phi i64 [ 0, %L40.lr.ph ], [ %value_phi7.7.lcssa.ph, %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa ]
  %value_phi3128.unr = phi double [ 1.000000e+00, %L40.lr.ph ], [ %value_phi6.7.lcssa.ph, %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa ]
  %value_phi2127.unr = phi double [ 0.000000e+00, %L40.lr.ph ], [ %value_phi5.7.lcssa.ph, %L35.L56_crit_edge.unr-lcssa.loopexit.unr-lcssa ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br label %L56

L56:                                              ; preds = %L35.L56_crit_edge.unr-lcssa, %top
  %value_phi2.lcssa = phi double [ 0.000000e+00, %top ], [ %value_phi5.lcssa.ph, %L35.L56_crit_edge.unr-lcssa ]
  %value_phi3.lcssa = phi double [ 1.000000e+00, %top ], [ %value_phi6.lcssa.ph, %L35.L56_crit_edge.unr-lcssa ]
  %117 = icmp slt i64 %"m::Int64", 0
  %.not96 = icmp eq i64 %"m::Int64", 0
  %118 = fmul double %value_phi2.lcssa, 1.000000e+00
  %119 = fadd double %value_phi3.lcssa, %118
  ret double %119
}

declare dso_local double @julia_log_3438(double) local_unnamed_addr #3

define hidden fastcc double @julia__recInf___3532(double %"a::Float64", double %"c::Float64", double %"w::Float64", i64 %"m::Int64", double %"\CF\B5::Float64") unnamed_addr {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(ptr %safepoint)
  fence syncscope("singlethread") seq_cst
  %1 = call double @llvm.fabs.f64(double %"\CF\B5::Float64")
  %2 = fcmp ule double %1, 1.000000e-01
  %3 = fsub double 1.000000e+00, %"c::Float64"
  %4 = fadd double %3, %"a::Float64"
  %5 = fadd double %4, %"\CF\B5::Float64"
  %6 = icmp slt i64 %"m::Int64", 0
  %.not86 = icmp eq i64 %"m::Int64", 0
  %7 = fneg double %"\CF\B5::Float64"
  %8 = call fastcc double @julia_G_3448(double 1.000000e+00, double %7)
  %9 = fmul double 1.000000e+00, %8
  %10 = call fastcc double @julia_P_3542(double %4, double %"\CF\B5::Float64", i64 %"m::Int64")
  %11 = fsub double 1.000000e+00, %"\CF\B5::Float64"
  %12 = call double @julia_gamma_3408(double %11)
  %13 = fdiv double %10, %12
  %14 = fsub double %9, %13
  %15 = fsub double %"c::Float64", %"a::Float64"
  %16 = call double @julia_gamma_3408(double %15)
  %17 = sitofp i64 %"m::Int64" to double
  %18 = fadd double %17, %"a::Float64"
  %19 = fadd double %18, %"\CF\B5::Float64"
  %20 = call double @julia_gamma_3408(double %19)
  %21 = fadd double %17, 1.000000e+00
  %22 = call double @julia_gamma_3408(double %21)
  %23 = fmul double %16, %20
  %24 = fmul double %23, %22
  %25 = fdiv double %14, %24
  %.not89 = icmp eq i64 %"m::Int64", 0
  %26 = add i64 %"m::Int64", 1
  %27 = sitofp i64 %26 to double
  %28 = call fastcc double @julia_G_3448(double %27, double %"\CF\B5::Float64")
  %29 = call double @julia_gamma_3408(double %19)
  %30 = fdiv double %28, %29
  %31 = call fastcc double @julia_G_3448(double %18, double %"\CF\B5::Float64")
  %32 = fadd double %27, %"\CF\B5::Float64"
  %33 = call double @julia_gamma_3408(double %32)
  %34 = fdiv double %31, %33
  %35 = fsub double %30, %34
  %36 = call double @julia_gamma_3408(double %15)
  %37 = fdiv double %35, %36
  %38 = call fastcc double @julia_G_3448(double %15, double %7)
  %39 = fneg double %"w::Float64"
  %40 = call double @julia_log_3438(double %39)
  %41 = fneg double %40
  %42 = fcmp une double %"\CF\B5::Float64", 0.000000e+00
  %43 = fsub double %15, %"\CF\B5::Float64"
  %44 = call double @julia_gamma_3408(double %43)
  %45 = fdiv double %41, %44
  %46 = fsub double %38, %45
  %47 = call double @julia_gamma_3408(double %32)
  %48 = call double @julia_gamma_3408(double %18)
  %49 = fmul double %47, %48
  %50 = fdiv double %46, %49
  %51 = fsub double %37, %50
  %52 = fmul double 1.000000e+00, %51
  %53 = fadd double %25, %52
  %54 = call double @julia_gamma_3408(double %"c::Float64")
  %.not92 = icmp eq i64 %"m::Int64", 0
  %55 = call double @julia___3420(double %"w::Float64", i64 %"m::Int64")
  %56 = fmul double %54, %53
  %57 = fmul double %56, 1.000000e+00
  %58 = fmul double %57, %55
  ret double %58
}

define hidden fastcc double @julia__recInf___3527(double %"a::Float64", double %"c::Float64", double %"w::Float64", i64 %"m::Int64", double %"\CF\B5::Float64") unnamed_addr {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(ptr %safepoint)
  fence syncscope("singlethread") seq_cst
  %1 = call double @julia_gamma_3408(double %"c::Float64")
  %2 = icmp slt i64 %"m::Int64", 0
  ret double 0.000000e+00
}

define hidden fastcc double @julia_BInf_3523(double %"a::Float64", double %"c::Float64", double %"win::Float64", i64 %"m::Int64", double %"\CF\B5::Float64") unnamed_addr {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %0 = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load ptr, ptr %0, align 8
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(ptr %safepoint)
  fence syncscope("singlethread") seq_cst
  %1 = call fastcc double @julia__recInf___3532(double %"a::Float64", double %"c::Float64", double %"win::Float64", i64 %"m::Int64", double %"\CF\B5::Float64")
  %2 = call fastcc double @julia__recInf___3527(double %"a::Float64", double %"c::Float64", double %"win::Float64", i64 %"m::Int64", double %"\CF\B5::Float64")
  %3 = fmul double %2, %"win::Float64"
  %4 = sitofp i64 %"m::Int64" to double
  %5 = fadd double %4, %"a::Float64"
  %6 = fsub double 1.000000e+00, %"c::Float64"
  %7 = fadd double %6, %"a::Float64"
  %8 = fadd double %7, %4
  %9 = add i64 %"m::Int64", 1
  br label %L20

L19:                                              ; preds = %L20
  ret double %37

L20:                                              ; preds = %L20, %top
  %value_phi38 = phi double [ %1, %top ], [ %36, %L20 ]
  %value_phi27 = phi double [ %3, %top ], [ %41, %L20 ]
  %value_phi16 = phi double [ %1, %top ], [ %37, %L20 ]
  %value_phi5 = phi i64 [ 0, %top ], [ %19, %L20 ]
  %10 = sitofp i64 %value_phi5 to double
  %11 = fadd double %5, %10
  %12 = fadd double %11, %"\CF\B5::Float64"
  %13 = fadd double %8, %10
  %14 = fadd double %13, %"\CF\B5::Float64"
  %15 = fmul double %12, %14
  %16 = add i64 %9, %value_phi5
  %17 = sitofp i64 %16 to double
  %18 = fadd double %17, %"\CF\B5::Float64"
  %19 = add i64 %value_phi5, 1
  %20 = sitofp i64 %19 to double
  %21 = fmul double %18, %20
  %22 = fdiv double %15, %21
  %23 = fmul double %22, %"win::Float64"
  %24 = fmul double %value_phi38, %23
  %25 = fmul double %11, %13
  %26 = fdiv double %25, %17
  %27 = fsub double %26, %11
  %28 = fsub double %27, %13
  %29 = fsub double %28, %"\CF\B5::Float64"
  %30 = fdiv double %15, %20
  %31 = fadd double %30, %29
  %32 = fmul double %value_phi27, %31
  %33 = fsub double %20, %"\CF\B5::Float64"
  %34 = fmul double %18, %33
  %35 = fdiv double %32, %34
  %36 = fadd double %24, %35
  %37 = fadd double %value_phi16, %36
  %38 = fmul double %33, %17
  %39 = fdiv double %25, %38
  %40 = fmul double %39, %"win::Float64"
  %41 = fmul double %value_phi27, %40
  %42 = call double @llvm.fabs.f64(double %36)
  %43 = call double @llvm.fabs.f64(double %37)
  %44 = fmul double %43, 8.000000e+00
  %45 = fmul double %44, 0x3CB0000000000000
  %46 = fcmp uge double %45, %42
  %47 = icmp ult i64 %value_phi5, 9223372036854775807
  %or.cond = select i1 %46, i1 %47, i1 false
  br i1 %or.cond, label %L19, label %L20

; uselistorder directives
  uselistorder double %37, { 0, 2, 1 }
}

declare double @__enzyme_autodiff(ptr, ...)

define double @df_binf(double %a, double %c, double %w, i64 %m, double %e) {
entry:
  %r = call double (ptr, ...) @__enzyme_autodiff(ptr @julia_BInf_3523, double %a, double %c, double %w, i64 %m, double %e)
  ret double %r
}

attributes #0 = { nofree "enzyme_ReadOnlyOrThrow" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { "enzyme_math"="powi" }
attributes #3 = { "enzyme_math"="log" }

; CHECK: define internal fastcc { double, double, double, double } @diffejulia_BInf_3523
