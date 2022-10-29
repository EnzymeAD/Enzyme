; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,inline,mem2reg,gvn,adce,instcombine,instsimplify,early-cse-memssa,simplifycfg,correlated-propagation,adce,loop-simplify,jump-threading,instsimplify,early-cse,simplifycfg"  -enzyme-preopt=false -S | FileCheck %s

%struct.n = type { double, %struct.n* }

; Function Attrs: noinline norecurse nounwind readonly uwtable
define dso_local double @sum_list(%struct.n* noalias readonly %node) #0 {
entry:
  %cmp6 = icmp eq %struct.n* %node, null
  br i1 %cmp6, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  ret double %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %val.08 = phi %struct.n* [ %1, %for.body ], [ %node, %entry ]
  %sum.07 = phi double [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %value = getelementptr inbounds %struct.n, %struct.n* %val.08, i64 0, i32 0
  %0 = load double, double* %value, align 8, !tbaa !2
  %add = fadd fast double %0, %sum.07
  %next = getelementptr inbounds %struct.n, %struct.n* %val.08, i64 0, i32 1
  %1 = load %struct.n*, %struct.n** %next, align 8, !tbaa !8
  %cmp = icmp eq %struct.n* %1, null
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind uwtable
define dso_local double @list_creator(double %x, i64 %n) #1 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %0 = bitcast i8* %call to %struct.n*
  %call2 = tail call fast double @sum_list(%struct.n* %0)
  ret double %call2

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %list.011 = phi %struct.n* [ null, %entry ], [ %1, %for.body ]
  %call = tail call noalias i8* @malloc(i64 16) #4
  %1 = bitcast i8* %call to %struct.n*
  %next = getelementptr inbounds i8, i8* %call, i64 8
  %2 = bitcast i8* %next to %struct.n**
  store %struct.n* %list.011, %struct.n** %2, align 8, !tbaa !8
  %value = bitcast i8* %call to double*
  store double %x, double* %value, align 8, !tbaa !2
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #2

; Function Attrs: noinline nounwind uwtable
define dso_local double @derivative(double %x, i64 %n) local_unnamed_addr #3 {
entry:
  %0 = tail call double (double (double, i64)*, ...) @__enzyme_autodiff(double (double, i64)* nonnull @list_creator, double %x, i64 %n)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_autodiff(double (double, i64)*, ...) #4

attributes #0 = { noinline norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !4, i64 0}
!3 = !{!"n", !4, i64 0, !7, i64 8}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"any pointer", !5, i64 0}
!8 = !{!3, !7, i64 8}
!9 = !{!7, !7, i64 0}

; CHECK: define dso_local double @derivative(double %x, i64 %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = shl i64 %n, 3
; CHECK-NEXT:   %mallocsize.i = add i64 %0, 8
; CHECK-NEXT:   %malloccall.i = call noalias nonnull i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:   %"call'mi_malloccache.i" = bitcast i8* %malloccall.i to i8**
; CHECK-NEXT:   %malloccall8.i = call noalias nonnull i8* @malloc(i64 %mallocsize.i) #4
; CHECK-NEXT:   %call_malloccache.i = bitcast i8* %malloccall8.i to i8**
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.cond.cleanup.i:                               ; preds = %for.body.i
; CHECK-NEXT:   call void @diffesum_list(%struct.n* nonnull %2, %struct.n* nonnull %"'ipc2.i", double 1.000000e+00) #4
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %iv.i = phi i64 [ %iv.next.i, %for.body.i ], [ 0, %entry ]
; CHECK-NEXT:   %1 = phi %struct.n* [ %"'ipc2.i", %for.body.i ], [ null, %entry ]
; CHECK-NEXT:   %list.011.i = phi %struct.n* [ %2, %for.body.i ], [ null, %entry ]
; CHECK-NEXT:   %iv.next.i = add nuw nsw i64 %iv.i, 1
; CHECK-NEXT:   %call.i = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16) #10
; CHECK-NEXT:   %"call'mi.i" = call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16) #10
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull align 1 dereferenceable(16) dereferenceable_or_null(16) %"call'mi.i", i8 0, i64 16, i1 false) #4
; CHECK-NEXT:   %"'ipc2.i" = bitcast i8* %"call'mi.i" to %struct.n*
; CHECK-NEXT:   %2 = bitcast i8* %call.i to %struct.n*
; CHECK-NEXT:   %"next'ipg.i" = getelementptr inbounds i8, i8* %"call'mi.i", i64 8
; CHECK-NEXT:   %next.i = getelementptr inbounds i8, i8* %call.i, i64 8
; CHECK-NEXT:   %"'ipc3.i" = bitcast i8* %"next'ipg.i" to %struct.n**
; CHECK-NEXT:   %3 = bitcast i8* %next.i to %struct.n**
; CHECK-NEXT:   store %struct.n* %1, %struct.n** %"'ipc3.i", align 8
; CHECK-NEXT:   %4 = getelementptr inbounds i8*, i8** %call_malloccache.i, i64 %iv.i
; CHECK-NEXT:   store i8* %call.i, i8** %4, align 8, !invariant.group !14
; CHECK-NEXT:   store %struct.n* %list.011.i, %struct.n** %3, align 8, !tbaa !8
; CHECK-NEXT:   %5 = getelementptr inbounds i8*, i8** %"call'mi_malloccache.i", i64 %iv.i
; CHECK-NEXT:   store i8* %"call'mi.i", i8** %5, align 8, !invariant.group !15
; CHECK-NEXT:   %value.i = bitcast i8* %call.i to double*
; CHECK-NEXT:   store double %x, double* %value.i, align 8, !tbaa !2
; CHECK-NEXT:   %exitcond.i = icmp eq i64 %iv.i, %n
; CHECK-NEXT:   br i1 %exitcond.i, label %for.cond.cleanup.i, label %for.body.i

; CHECK: invertfor.body.i:                                 ; preds = %incinvertfor.body.i, %for.cond.cleanup.i
; CHECK-NEXT:   %"x'de.i.0" = phi double [ 0.000000e+00, %for.cond.cleanup.i ], [ %9, %incinvertfor.body.i ]
; CHECK-NEXT:   %"iv'ac.i.0" = phi i64 [ %n, %for.cond.cleanup.i ], [ %13, %incinvertfor.body.i ]
; CHECK-NEXT:   %6 = getelementptr inbounds i8*, i8** %"call'mi_malloccache.i", i64 %"iv'ac.i.0"
; CHECK-NEXT:   %7 = load i8*, i8** %6, align 8, !invariant.group !15
; CHECK-NEXT:   %"value'ipc_unwrap.i" = bitcast i8* %7 to double*
; CHECK-NEXT:   %8 = load double, double* %"value'ipc_unwrap.i", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"value'ipc_unwrap.i", align 8
; CHECK-NEXT:   %9 = fadd fast double %"x'de.i.0", %8
; CHECK-NEXT:   call void @free(i8* nonnull %7) #4
; CHECK-NEXT:   %10 = getelementptr inbounds i8*, i8** %call_malloccache.i, i64 %"iv'ac.i.0"
; CHECK-NEXT:   %11 = load i8*, i8** %10, align 8, !invariant.group !14
; CHECK-NEXT:   call void @free(i8* %11) #4
; CHECK-NEXT:   %12 = icmp eq i64 %"iv'ac.i.0", 0
; CHECK-NEXT:   br i1 %12, label %diffelist_creator.exit, label %incinvertfor.body.i

; CHECK: incinvertfor.body.i:                              ; preds = %invertfor.body.i
; CHECK-NEXT:   %13 = add nsw i64 %"iv'ac.i.0", -1
; CHECK-NEXT:   br label %invertfor.body.i

; CHECK: diffelist_creator.exit:                           ; preds = %invertfor.body.i
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall.i) #4
; CHECK-NEXT:   call void @free(i8* nonnull %malloccall8.i) #4
; CHECK-NEXT:   ret double %9
; CHECK-NEXT: }



; CHECK: define internal void @diffesum_list(%struct.n* noalias readonly %node, %struct.n* %"node'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp6 = icmp eq %struct.n* %node, null
; CHECK-NEXT:   br i1 %cmp6, label %invertentry, label %for.body

; CHECK: for.body:                                         ; preds = %entry, %__enzyme_exponentialallocation.exit
; CHECK-NEXT:   %0 = phi i8* [ %11, %__enzyme_exponentialallocation.exit ], [ null, %entry ]
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %__enzyme_exponentialallocation.exit ], [ 0, %entry ]
; CHECK-NEXT:   %1 = phi %struct.n* [ %"'ipl", %__enzyme_exponentialallocation.exit ], [ %"node'", %entry ]
; CHECK-NEXT:   %val.08 = phi %struct.n* [ %14, %__enzyme_exponentialallocation.exit ], [ %node, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %2 = and i64 %iv.next, 1
; CHECK-NEXT:   %3 = icmp ne i64 %2, 0
; CHECK-NEXT:   %4 = call i64 @llvm.ctpop.i64(i64 %iv.next) #4, !range !22
; CHECK-NEXT:   %5 = icmp ult i64 %4, 3
; CHECK-NEXT:   %6 = and i1 %5, %3
; CHECK-NEXT:   br i1 %6, label %grow.i, label %__enzyme_exponentialallocation.exit

; CHECK: grow.i:                                           ; preds = %for.body
; CHECK-NEXT:   %7 = call i64 @llvm.ctlz.i64(i64 %iv.next, i1 true) #4, !range !23
; CHECK-NEXT:   %8 = sub nuw nsw i64 64, %7
; CHECK-NEXT:   %9 = shl i64 8, %8
; CHECK-NEXT:   %10 = call i8* @realloc(i8* %0, i64 %9) #4
; CHECK-NEXT:   br label %__enzyme_exponentialallocation.exit

; CHECK: __enzyme_exponentialallocation.exit:              ; preds = %for.body, %grow.i
; CHECK-NEXT:   %11 = phi i8* [ %10, %grow.i ], [ %0, %for.body ]
; CHECK-NEXT:   %12 = bitcast i8* %11 to %struct.n**
; CHECK-NEXT:   %13 = getelementptr inbounds %struct.n*, %struct.n** %12, i64 %iv
; CHECK-NEXT:   store %struct.n* %1, %struct.n** %13, align 8, !invariant.group !24
; CHECK-NEXT:   %"next'ipg" = getelementptr inbounds %struct.n, %struct.n* %1, i64 0, i32 1
; CHECK-NEXT:   %next = getelementptr inbounds %struct.n, %struct.n* %val.08, i64 0, i32 1
; CHECK-NEXT:   %"'ipl" = load %struct.n*, %struct.n** %"next'ipg", align 8, !tbaa !8
; CHECK-NEXT:   %14 = load %struct.n*, %struct.n** %next, align 8, !tbaa !8
; CHECK-NEXT:   %cmp = icmp eq %struct.n* %14, null
; CHECK-NEXT:   br i1 %cmp, label %invertfor.body, label %for.body

; CHECK: invertentry:                                      ; preds = %entry, %invertfor.body.preheader
; CHECK-NEXT:   ret void

; CHECK: invertfor.body.preheader:                         ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %11)
; CHECK-NEXT:   br label %invertentry

; CHECK: invertfor.body:                                   ; preds = %__enzyme_exponentialallocation.exit, %incinvertfor.body
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %20, %incinvertfor.body ], [ %iv, %__enzyme_exponentialallocation.exit ]
; CHECK-NEXT:   %15 = getelementptr inbounds %struct.n*, %struct.n** %12, i64 %"iv'ac.0"
; CHECK-NEXT:   %16 = load %struct.n*, %struct.n** %15, align 8, !invariant.group !24
; CHECK-NEXT:   %"value'ipg_unwrap" = getelementptr inbounds %struct.n, %struct.n* %16, i64 0, i32 0
; CHECK-NEXT:   %17 = load double, double* %"value'ipg_unwrap", align 8
; CHECK-NEXT:   %18 = fadd fast double %17, %differeturn
; CHECK-NEXT:   store double %18, double* %"value'ipg_unwrap", align 8
; CHECK-NEXT:   %19 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %19, label %invertfor.body.preheader, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %20 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body
; CHECK-NEXT: }