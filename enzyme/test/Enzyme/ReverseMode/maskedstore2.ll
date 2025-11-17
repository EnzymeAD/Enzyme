; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -dce -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,dce,instsimplify)" -S | FileCheck %s

declare void @llvm.masked.store.v2f64.p0v2f64  (<2 x double>, <2 x double>*, i32, <2 x i1>)

; Function Attrs: nounwind uwtable
define dso_local void @loader(i1 %cmp, <2 x double>* %ptr, <2 x i64> %mask0, <2 x double> %val) {
entry:
  br i1 %cmp, label %t, label %exit

t:
  %mask = icmp eq <2 x i64> %mask0, <i64 0, i64 1>
  call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask)
  br label %exit
  
exit:
  ret void
}


; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define <2 x double> @dloader(i8* %ptr, i8* %dptr, <2 x i64> %mask, <2 x double> %other) {
entry:
  %res = tail call <2 x double> (...) @__enzyme_autodiff.f64(void (i1, <2 x double>*, <2 x i64>, <2 x double>)* @loader, i1 true, i8* %ptr, i8* %dptr, <2 x i64> %mask, <2 x double> %other)
  ret <2 x double> %res
}

declare <2 x double> @__enzyme_autodiff.f64(...) 


; CHECK: define internal { <2 x double> } @diffeloader(i1 %cmp, <2 x double>* writeonly %ptr, <2 x double>* %"ptr'", <2 x i64> %mask0, <2 x double> %val)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"val'de" = alloca <2 x double>
; CHECK-NEXT:   store <2 x double> zeroinitializer, <2 x double>* %"val'de"
; CHECK-NEXT:   br i1 %cmp, label %t, label %invertentry

; CHECK: t:                                                ; preds = %entry
; CHECK-NEXT:   %mask = icmp eq <2 x i64> %mask0, <i64 0, i64 1>
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   %mask_unwrap = icmp eq <2 x i64> %mask0, <i64 0, i64 1>
; CHECK-NEXT:   %0 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %"ptr'", i32 16, <2 x i1> %mask_unwrap, <2 x double> zeroinitializer)
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> zeroinitializer, <2 x double>* %"ptr'", i32 16, <2 x i1> %mask_unwrap)
; CHECK-NEXT:   %1 = load <2 x double>, <2 x double>* %"val'de"
; CHECK-NEXT:   %2 = fadd fast <2 x double> %1, %0
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %2, <2 x double>* %"val'de", i32 16, <2 x i1> %mask_unwrap)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry, %t
; CHECK-NEXT:   %3 = load <2 x double>, <2 x double>* %"val'de"
; CHECK-NEXT:   %4 = insertvalue { <2 x double> } undef, <2 x double> %3, 0
; CHECK-NEXT:   ret { <2 x double> } %4
; CHECK-NEXT: }
