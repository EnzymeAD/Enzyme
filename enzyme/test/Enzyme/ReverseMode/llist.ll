; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -gvn -adce -instcombine -instsimplify -early-cse -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,gvn,adce,instcombine,instsimplify,early-cse,%simplifycfg,correlated-propagation,adce,jump-threading,instsimplify)" -S | FileCheck %s

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

; CHECK: define internal { double } @diffelist_creator(double %x, i64 %n, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %[[n8:.+]] = shl i64 %n, 3
; CHECK-NEXT:  %mallocsize = add i64 %[[n8]], 8
; CHECK-NEXT:  %[[mallocp:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:  %[[callpcache:.+]] = bitcast i8* %[[mallocp]] to i8**
; CHECK-NEXT:  %[[malloc1:.+]] = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:  %call_malloccache = bitcast i8* %[[malloc1:.+]] to i8**
; CHECK-NEXT:  br label %for.body

; CHECK:[[invertforcondcleanup:.+]]:
; CHECK-NEXT:  call void @diffesum_list(%struct.n* nonnull %[[thisbc:.+]], %struct.n* nonnull %[[dstructncast:.+]], double %differeturn)
; CHECK-NEXT:  br label %invertfor.body

; CHECK:for.body:                                       ; preds = %for.body, %entry
; CHECK-NEXT:  %[[iv:.+]] = phi i64 [ %[[ivnext:.+]], %for.body ], [ 0, %entry ]
; CHECK-NEXT:  %[[structtostore:.+]] = phi %struct.n* [ %[[dstructncast]], %for.body ], [ null, %entry ]
; CHECK-NEXT:  %list.011 = phi %struct.n* [ %[[thisbc]], %for.body ], [ null, %entry ]
; CHECK-NEXT:  %[[ivnext]] = add nuw nsw i64 %[[iv]], 1

; CHECK-NEXT:  %"call'mi" = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)
; CHECK-NEXT:  call void @llvm.memset.p0i8.i64(i8* {{(noundef )?}}nonnull {{(align 1 )?}}dereferenceable(16) dereferenceable_or_null(16) %"call'mi", i8 0, i64 16, {{(i32 1, )?}}i1 false)
; CHECK-NEXT:  %call = tail call noalias nonnull dereferenceable(16) dereferenceable_or_null(16) i8* @malloc(i64 16)

; CHECK-NEXT:  %[[dstructncast]] = bitcast i8* %"call'mi" to %struct.n*
; CHECK-NEXT:  %[[thisbc]] = bitcast i8* %call to %struct.n*
; CHECK-NEXT:  %[[nextipgi:.+]] = getelementptr inbounds i8, i8* %"call'mi", i64 8
; CHECK-NEXT:  %next = getelementptr inbounds i8, i8* %call, i64 8
; CHECK-NEXT:  %[[dstruct1:.+]] = bitcast i8* %[[nextipgi]] to %struct.n**
; CHECK-NEXT:  %[[fbc:.+]] = bitcast i8* %next to %struct.n**

; CHECK-NEXT:  store %struct.n* %[[structtostore]], %struct.n** %[[dstruct1]]
; CHECK-NEXT:  %[[callcachegep:.+]] = getelementptr inbounds i8*, i8** %call_malloccache, i64 %[[iv]]
; CHECK-NEXT:  store i8* %call, i8** %[[callcachegep]]

; CHECK-NEXT:  store %struct.n* %list.011, %struct.n** %[[fbc]], align 8, !tbaa !8

; CHECK-NEXT:  %[[callpcachegep:.+]] = getelementptr inbounds i8*, i8** %[[callpcache]], i64 %[[iv]]
; CHECK-NEXT:  store i8* %"call'mi", i8** %[[callpcachegep]]

; CHECK-NEXT:  %value = bitcast i8* %call to double*
; CHECK-NEXT:  store double %x, double* %value, align 8, !tbaa !2
; CHECK-NEXT:  %[[exitcond:.+]] = icmp eq i64 %[[iv]], %n
; CHECK-NEXT:  br i1 %[[exitcond]], label %[[invertforcondcleanup]], label %for.body

; CHECK: invertentry: 
; CHECK-NEXT:  %[[res6:.+]] = insertvalue { double } {{(undef|poison)}}, double %[[add:.+]], 0
; CHECK-NEXT:  call void @free(i8* nonnull %[[mallocp]])
; CHECK-NEXT:  call void @free(i8* nonnull %[[malloc1]])
; CHECK-NEXT:  ret { double } %[[res6]]

; CHECK:invertfor.body:
; CHECK-NEXT:  %"x'de.0" = phi double [ 0.000000e+00, %[[invertforcondcleanup]] ], [ %[[add]], %incinvertfor.body ]
; CHECK-NEXT:  %[[antivar:.+]] = phi i64 [ %n, %[[invertforcondcleanup]] ], [ %[[sub:.+]], %incinvertfor.body ]
; CHECK-NEXT:  %[[gep:.+]] = getelementptr inbounds i8*, i8** %"call'mi_malloccache", i64 %[[antivar]]
; CHECK-NEXT:  %[[loadcache:.+]] = load i8*, i8** %[[gep]]
; CHECK-NEXT:  %[[ccast:.+]] = bitcast i8* %[[loadcache]] to double*
; CHECK-NEXT:  %[[load:.+]] = load double, double* %[[ccast]]
; this store is optional and could get removed by DCE
; CHECK-NEXT:  store double 0.000000e+00, double* %[[ccast]]
; CHECK-NEXT:  %[[add]] = fadd fast double %"x'de.0", %[[load]]
; CHECK-NEXT:  call void @free(i8* nonnull %[[loadcache]])
; CHECK-NEXT:  %[[gepcall:.+]] = getelementptr inbounds i8*, i8** %call_malloccache, i64 %[[antivar]]
; CHECK-NEXT:  %[[loadprefree:.+]] = load i8*, i8** %[[gepcall]]
; CHECK-NEXT:  call void @free(i8* %[[loadprefree]]) 
; CHECK-NEXT:  %[[cmp:.+]] = icmp eq i64 %[[antivar]], 0
; CHECK-NEXT:  br i1 %[[cmp:.+]], label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:
; CHECK-NEXT:  %[[sub]] = add nsw i64 %[[antivar]], -1
; CHECK-NEXT:  br label %invertfor.body
