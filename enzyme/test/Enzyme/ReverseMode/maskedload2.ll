; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -dce -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,dce,instsimplify)" -S | FileCheck %s

declare <2 x double>  @llvm.masked.load.v2f64.p0v2f64  (<2 x double>*, i32, <2 x i1>, <2 x double>)

; Function Attrs: nounwind uwtable
define dso_local <2 x double> @loader(i1 %cmp, <2 x double>* %ptr, <2 x i64> %mask0, <2 x double> %other) {
entry:
  br i1 %cmp, label %t, label %exit

t:
  %mask = icmp eq <2 x i64> %mask0, <i64 0, i64 1>
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %ptr, i32 16, <2 x i1> %mask, <2 x double> %other)
  br label %exit

exit:
  %p = phi <2 x double> [ %res, %t ], [ zeroinitializer, %entry ]
  ret <2 x double> %p
}


; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define <2 x double> @dloader(i8* %ptr, i8* %dptr, <2 x i64> %mask, <2 x double> %other) {
entry:
  %res = tail call <2 x double> (...) @__enzyme_autodiff.f64(<2 x double> (i1, <2 x double>*, <2 x i64>, <2 x double>)* @loader, i1 true, i8* %ptr, i8* %dptr, <2 x i64> %mask, <2 x double> %other)
  ret <2 x double> %res
}

declare <2 x double> @__enzyme_autodiff.f64(...) 

; CHECK: define internal { <2 x double> } @diffeloader(i1 %cmp, <2 x double>* readonly %ptr, <2 x double>* %"ptr'", <2 x i64> %mask0, <2 x double> %other, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"other'de" = alloca <2 x double>
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %"other'de"
; CHECK-NEXT:   br i1 %cmp, label %t, label %invertexit

; CHECK: t:                                                ; preds = %entry
; CHECK-NEXT:   br label %invertexit

; CHECK: invertentry:                                      ; preds = %invertexit, %invertt
; CHECK-NEXT:   %0 = load <2 x double>, <2 x double>* %"other'de"
; CHECK-NEXT:   %1 = insertvalue { <2 x double> } {{(undef|poison)?}}, <2 x double> %0, 0
; CHECK-NEXT:   ret { <2 x double> } %1

; CHECK: invertt:                                          ; preds = %invertexit
; CHECK-NEXT:   %mask_unwrap = icmp eq <2 x i64> %mask0, <i64 0, i64 1>
; CHECK-NEXT:   %2 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %"ptr'", i32 16, <2 x i1> %mask_unwrap, <2 x double> zeroinitializer)
; CHECK-NEXT:   %3 = fadd fast <2 x double> %2, %7
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %3, <2 x double>* %"ptr'", i32 16, <2 x i1> %mask_unwrap)
; CHECK-NEXT:   %4 = xor <2 x i1> %mask_unwrap, <i1 true, i1 true>
; CHECK-NEXT:   %5 = load <2 x double>, <2 x double>* %"other'de"
; CHECK-NEXT:   %6 = fadd fast <2 x double> %5, %7
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %6, <2 x double>* %"other'de", i32 16, <2 x i1> %4)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertexit:                                       ; preds = %entry, %t
; CHECK-NEXT:   %7 = select {{(fast )?}}i1 %cmp, <2 x double> %differeturn, <2 x double> zeroinitializer
; CHECK-NEXT:   br i1 %cmp, label %invertt, label %invertentry
; CHECK-NEXT: }
