; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -sroa -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,sroa,early-cse,instsimplify,%simplifycfg)" -S | FileCheck %s

@.str = private unnamed_addr constant [12 x i8] c"x=%f xp=%f\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local double @allocateAndSet(double** nocapture %arrayp, double returned %x, i32 %n) local_unnamed_addr #0 {
entry:
  %conv = zext i32 %n to i64
  %mul = shl nuw nsw i64 %conv, 3
  %call = tail call i8* @__rust_alloc_zeroed(i64 %mul, i64 4)
  %0 = bitcast double** %arrayp to i8**
  store i8* %call, i8** %0, align 8, !tbaa !2
  %arrayidx = getelementptr inbounds i8, i8* %call, i64 24
  %1 = bitcast i8* %arrayidx to double*
  store double %x, double* %1, align 8, !tbaa !6
  ret double %x
}

declare dso_local noalias i8* @__rust_alloc_zeroed(i64, i64)

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @get(double* nocapture readonly %x, i32 %i) local_unnamed_addr #2 {
entry:
  %idxprom = zext i32 %i to i64
  %arrayidx = getelementptr inbounds double, double* %x, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8, !tbaa !6
  ret double %0
}

; Function Attrs: nounwind uwtable
define dso_local double @function(double %x, i32 %n) #3 {
entry:
  %array = alloca double*, align 8
  %0 = bitcast double** %array to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) 
  %call = call fast double @allocateAndSet(double** nonnull %array, double %x, i32 %n)
  %1 = load double*, double** %array, align 8, !tbaa !2
  %call1 = tail call fast double @get(double* %1, i32 3)
  %add = fadd fast double %call1, %call
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #5
  ret double %add
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x, i32 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (double (double, i32)*, ...) @__enzyme_autodiff(double (double, i32)* nonnull @function, double %x, i32 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i32)*, ...) #5


attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}

; CHECK: define internal { i8*, i8* } @augmented_allocateAndSet(double** nocapture %arrayp, double** nocapture %"arrayp'", double %x, i32 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %conv = zext i32 %n to i64
; CHECK-NEXT:   %mul = shl nuw nsw i64 %conv, 3
; CHECK-NEXT:   %"call'mi" = tail call noalias nonnull i8* @__rust_alloc_zeroed(i64 %mul, i64 4) 
; CHECK-NEXT:   %call = tail call i8* @__rust_alloc_zeroed(i64 %mul, i64 4) 
; CHECK-NEXT:   %"'ipc" = bitcast double** %"arrayp'" to i8**
; CHECK-NEXT:   %0 = bitcast double** %arrayp to i8**
; CHECK-NEXT:   store i8* %"call'mi", i8** %"'ipc", align 8
; CHECK-NEXT:   store i8* %call, i8** %0, align 8
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i8, i8* %call, i64 24
; CHECK-NEXT:   %1 = bitcast i8* %arrayidx to double*
; CHECK-NEXT:   store double %x, double* %1, align 8
; CHECK-NEXT:   %.fca.0.insert = insertvalue { i8*, i8* } {{(undef|poison)}}, i8* %"call'mi", 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue { i8*, i8* } %.fca.0.insert, i8* %call, 1
; CHECK-NEXT:   ret { i8*, i8* } %.fca.1.insert
; CHECK-NEXT: }

; CHECK: define internal { double } @diffeallocateAndSet(double** nocapture %arrayp, double** nocapture %"arrayp'", double %x, i32 %n, double %differeturn, { i8*, i8* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %conv = zext i32 %n to i64
; CHECK-NEXT:   %mul = shl nuw nsw i64 %conv, 3
; CHECK-NEXT:   %"call'mi" = extractvalue { i8*, i8* } %tapeArg, 0
; CHECK-NEXT:   %call = extractvalue { i8*, i8* } %tapeArg, 1
; CHECK-NEXT:   %"arrayidx'ipg" = getelementptr inbounds i8, i8* %"call'mi", i64 24
; CHECK-NEXT:   %"'ipc" = bitcast i8* %"arrayidx'ipg" to double*
; CHECK-NEXT:   %0 = load double, double* %"'ipc", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"'ipc", align 8
; CHECK-NEXT:   %1 = fadd fast double %differeturn, %0
; CHECK-NEXT:   call void @__rust_dealloc(i8* nonnull %"call'mi", i64 %mul, i64 4)
; CHECK-NEXT:   call void @__rust_dealloc(i8* %call, i64 %mul, i64 4)
; CHECK-NEXT:   %2 = insertvalue { double } {{(undef|poison)}}, double %1, 0
; CHECK-NEXT:   ret { double } %2
; CHECK-NEXT: }
