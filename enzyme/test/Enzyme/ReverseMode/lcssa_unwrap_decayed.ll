; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s

; The reverse of L62.i needs the address of the double array, which is reached
; by loading an index out of one array and using it to index another. That
; address is a julia decayed pointer, so it may not be stored to the tape, and
; the block defining it is not reachable from the scope the reverse pass asks
; for it in, so it cannot be looked up either. Enzyme used to assert with
; "undef value upon lcssa"; the whole chain has to be rebuilt in the reverse.

source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

define "enzyme_type"="{[-1]:Float@double}" double @indexed_by_load({ {} addrspace(10)*, {} addrspace(10)*, [1 x {} addrspace(10)*] } "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [16,0,0,0]:Pointer, [16,0,0,0,-1]:Float@double}" %0) #1 {
entry:
  %.fca.2.0.extract = extractvalue { {} addrspace(10)*, {} addrspace(10)*, [1 x {} addrspace(10)*] } %0, 2, 0
  br i1 false, label %exit, label %L8.i.lr.ph

L8.i.lr.ph:                                       ; preds = %entry
  br label %L8.i

L8.i:                                             ; preds = %L34.i, %L8.i.lr.ph
  br label %pass.i

L23.i:                                            ; preds = %pass.i
  %1 = addrspacecast {} addrspace(10)* %getfield6.i to i32 addrspace(13)* addrspace(11)*
  %arrayptr19.i16 = load i32 addrspace(13)*, i32 addrspace(13)* addrspace(11)* %1, align 16
  %arrayref20.i = load i32, i32 addrspace(13)* %arrayptr19.i16, align 4
  %2 = zext i32 %arrayref20.i to i64
  %3 = add nsw i64 %2, 0
  %4 = getelementptr inbounds { {} addrspace(10)*, i32 }, { {} addrspace(10)*, i32 } addrspace(13)* null, i64 %3
  %arrayref25.i = load { {} addrspace(10)*, i32 }, { {} addrspace(10)*, i32 } addrspace(13)* %4, align 8
  br label %pass24.i

L34.i:                                            ; preds = %pass.i
  br i1 false, label %exit, label %L8.i

L62.i:                                            ; No predecessors!
  %5 = addrspacecast {} addrspace(10)* %arrayref32.i to double addrspace(13)* addrspace(11)*
  %arrayptr44.i22 = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %5, align 8
  %value_phi50.i54 = load double, double addrspace(13)* %arrayptr44.i22, align 8
  %6 = fadd double 0.000000e+00, %value_phi50.i54
  br label %L90.i

L90.i:                                            ; preds = %L62.i
  br label %exit

pass.i:                                           ; preds = %L8.i
  %getfield6.i = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* null unordered, align 8
  br i1 false, label %L34.i, label %L23.i

pass24.i:                                         ; preds = %L23.i
  %7 = extractvalue { {} addrspace(10)*, i32 } %arrayref25.i, 1
  %8 = zext i32 %7 to i64
  %9 = add nsw i64 %8, 0
  %10 = addrspacecast {} addrspace(10)* %.fca.2.0.extract to {} addrspace(10)* addrspace(13)* addrspace(11)*
  %arrayptr29.i19 = load {} addrspace(10)* addrspace(13)*, {} addrspace(10)* addrspace(13)* addrspace(11)* %10, align 8
  %11 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %arrayptr29.i19, i64 %9
  %arrayref32.i = load {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %11, align 8
  ret double 0.000000e+00

exit:                            ; preds = %L90.i, %L34.i, %entry
  %value_phi106.i = phi double [ 0.000000e+00, %entry ], [ %6, %L90.i ], [ 0.000000e+00, %L34.i ]
  ret double %value_phi106.i
}

declare double @__enzyme_autodiff(...)

define double @dsquare({ {} addrspace(10)*, {} addrspace(10)*, [1 x {} addrspace(10)*] } %q, { {} addrspace(10)*, {} addrspace(10)*, [1 x {} addrspace(10)*] } %dq) {
entry:
  %call = tail call double (...) @__enzyme_autodiff(i8* bitcast (double ({ {} addrspace(10)*, {} addrspace(10)*, [1 x {} addrspace(10)*] })* @indexed_by_load to i8*), metadata !"enzyme_dup", { {} addrspace(10)*, {} addrspace(10)*, [1 x {} addrspace(10)*] } zeroinitializer, { {} addrspace(10)*, {} addrspace(10)*, [1 x {} addrspace(10)*] } zeroinitializer)
  ret double 0.000000e+00
}

attributes #1 = { "enzyme_ta_norecur" }

; CHECK: define internal void @diffeindexed_by_load(
; CHECK: invertL62.i:
; CHECK-NEXT:   %13 = load double, double* %"'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'de", align 8
; CHECK-NEXT:   %14 = load double, double* %"value_phi50.i54'de", align 8
; CHECK-NEXT:   %15 = fadd fast double %14, %13
; CHECK-NEXT:   store double %15, double* %"value_phi50.i54'de", align 8
; CHECK-NEXT:   %16 = load double, double* %"value_phi50.i54'de", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"value_phi50.i54'de", align 8
; CHECK-NEXT:   %"'ipc1_unwrap" = addrspacecast {} addrspace(10)* %".fca.2.0.extract'ipev" to {} addrspace(10)* addrspace(13)* addrspace(11)*
; CHECK-NEXT:   %"arrayptr29.i19'il_phi_unwrap" = load {} addrspace(10)* addrspace(13)*, {} addrspace(10)* addrspace(13)* addrspace(11)* %"'ipc1_unwrap", align 8
; CHECK-NEXT:   %getfield6.i_unwrap = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* null unordered, align 8
; CHECK-NEXT:   %_unwrap = addrspacecast {} addrspace(10)* %getfield6.i_unwrap to i32 addrspace(13)* addrspace(11)*
; CHECK-NEXT:   %arrayptr19.i16_unwrap = load i32 addrspace(13)*, i32 addrspace(13)* addrspace(11)* %_unwrap, align 16
; CHECK-NEXT:   %arrayref20.i_unwrap = load i32, i32 addrspace(13)* %arrayptr19.i16_unwrap, align 4
; CHECK-NEXT:   %_unwrap2 = zext i32 %arrayref20.i_unwrap to i64
; CHECK-NEXT:   %_unwrap3 = add nsw i64 %_unwrap2, 0
; CHECK-NEXT:   %_unwrap4 = getelementptr inbounds { {} addrspace(10)*, i32 }, { {} addrspace(10)*, i32 } addrspace(13)* null, i64 %_unwrap3
; CHECK-NEXT:   %arrayref25.i_unwrap = load { {} addrspace(10)*, i32 }, { {} addrspace(10)*, i32 } addrspace(13)* %_unwrap4, align 8
; CHECK-NEXT:   %_unwrap5 = extractvalue { {} addrspace(10)*, i32 } %arrayref25.i_unwrap, 1
; CHECK-NEXT:   %_unwrap6 = zext i32 %_unwrap5 to i64
; CHECK-NEXT:   %_unwrap7 = add nsw i64 %_unwrap6, 0
; CHECK-NEXT:   %"'ipg_unwrap" = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %"arrayptr29.i19'il_phi_unwrap", i64 %_unwrap7
; CHECK-NEXT:   %"arrayref32.i'il_phi_unwrap" = load {} addrspace(10)*, {} addrspace(10)* addrspace(13)* %"'ipg_unwrap", align 8
; CHECK-NEXT:   %"'ipc_unwrap" = addrspacecast {} addrspace(10)* %"arrayref32.i'il_phi_unwrap" to double addrspace(13)* addrspace(11)*
; CHECK-NEXT:   %"arrayptr44.i22'il_phi_unwrap" = load double addrspace(13)*, double addrspace(13)* addrspace(11)* %"'ipc_unwrap", align 8
; CHECK-NEXT:   %17 = load double, double addrspace(13)* %"arrayptr44.i22'il_phi_unwrap", align 8
; CHECK-NEXT:   %18 = fadd fast double %17, %16
; CHECK-NEXT:   store double %18, double addrspace(13)* %"arrayptr44.i22'il_phi_unwrap", align 8
; CHECK-NEXT:   ret void
