; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -gvn -early-cse -instsimplify -simplifycfg -adce -S | FileCheck %s

define void @foo(float* noalias %out, float* noalias %in, i64* %x2.i.i, i1 %a9) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body.i.i.preheader.i.i.i.i.i.i59, %for.cond.cleanup4
  %s1.0235 = phi i64 [ 0, %entry ], [ %inc16, %for.cond.cleanup4 ]
  br label %for.body2

for.body2:                                       ; preds = %for.body, %_ZNK11OuterStruct11InnerStruct5sizexEv.exit.i
  %j = phi i64 [ %nextj, %merge ], [ 0, %for.body ]
  br i1 %a9, label %cond.false, label %merge

cond.false:                                   ; preds = %for.body.i
  %a14 = load i64, i64* %x2.i.i, align 8
  store i64 1, i64* %x2.i.i
  br label %merge

merge:    ; preds = %cond.false.i.i, %cond.true.i.i
  %cond = phi i64 [ 2, %for.body2 ], [ %a14, %cond.false ]
  %nextj = add i64 %j, 1
  %cmp = icmp eq i64 %nextj, 7
  br i1 %cmp, label %_ZNK11OuterStruct4sizeEv.exit, label %for.body2

_ZNK11OuterStruct4sizeEv.exit:                    ; preds = %_ZNK11OuterStruct11InnerStruct5sizexEv.exit.i, %for.body
  %s.0.lcssa.i = phi i64 [ %cond, %merge ]
  %cmp3.not233 = icmp eq i64 %s.0.lcssa.i, 0
  br i1 %cmp3.not233, label %for.cond.cleanup4, label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %_ZNK11OuterStruct4sizeEv.exit, %for.cond6.preheader.split.split
  %i = phi i64 [ %nexti, %for.cond6.preheader ], [ 0, %_ZNK11OuterStruct4sizeEv.exit ]
  %a17 = load float, float* %in, align 8
  %sq = fmul float %a17, %a17
  store float %sq, float* %out, align 8
  %nexti = add nuw i64 %i, 1
  %cmp3.not = icmp eq i64 %nexti, %s.0.lcssa.i
  br i1 %cmp3.not, label %for.cond.cleanup4, label %for.cond6.preheader

for.cond.cleanup4:                                ; preds = %for.cond6.preheader.split.split, %_ZNK11OuterStruct4sizeEv.exit
  %inc16 = add nuw nsw i64 %s1.0235, 1
  %cmp.not = icmp eq i64 %inc16, 10
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  store float 0.000000e+00, float* %in, align 8
  ret void
}

; Function Attrs: nounwind uwtable
define void @caller(float* %in, float* %din, float* %outstr, float* %d_outstr, i64* %l, i1 %s) local_unnamed_addr {
entry:
  call void @__enzyme_autodiff(i8* bitcast (void (float*, float*, i64*, i1)* @foo to i8*), float* %in, float* %din, float* %outstr, float* %d_outstr, i64* %l, i1 %s) #11
  ret void
}

declare void @__enzyme_autodiff(i8*, float*, float*, float*, float*, i64* %l, i1)

; CHECK: define internal void @diffefoo(float* noalias %out, float* %"out'", float* noalias %in, float* %"in'", i64* %x2.i.i, i1 %a9) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %a17_malloccache = bitcast i8* %malloccall to float**
; CHECK-NEXT:   %malloccall9 = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %cond.lcssa_malloccache = bitcast i8* %malloccall9 to i64*
; CHECK-NEXT:   %malloccall13 = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* @malloc(i64 80)
; CHECK-NEXT:   %s.0.lcssa.i_malloccache = bitcast i8* %malloccall13 to i64*
; CHECK-NEXT:   br label %for.body

; CHECK: for.body:                                         ; preds = %for.cond.cleanup4, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.cond.cleanup4 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   br label %for.body2

; CHECK: for.body2:                                        ; preds = %merge, %for.body
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %merge ], [ 0, %for.body ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   br i1 %a9, label %cond.false, label %merge

; CHECK: cond.false:                                       ; preds = %for.body2
; CHECK-NEXT:   %a14 = load i64, i64* %x2.i.i, align 8
; CHECK-NEXT:   store i64 1, i64* %x2.i.i, align 4
; CHECK-NEXT:   br label %merge

; CHECK: merge:                                            ; preds = %cond.false, %for.body2
; CHECK-NEXT:   %cond = phi i64 [ 2, %for.body2 ], [ %a14, %cond.false ]
; CHECK-NEXT:   %cmp = icmp eq i64 %iv.next2, 7
; CHECK-NEXT:   br i1 %cmp, label %_ZNK11OuterStruct4sizeEv.exit, label %for.body2

; CHECK: _ZNK11OuterStruct4sizeEv.exit:                    ; preds = %merge
; CHECK-NEXT:   %0 = getelementptr inbounds i64, i64* %cond.lcssa_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %cond, i64* %0, align 8, !invariant.group !0
; CHECK-NEXT:   %1 = getelementptr inbounds i64, i64* %s.0.lcssa.i_malloccache, i64 %iv
; CHECK-NEXT:   store i64 %cond, i64* %1, align 8, !invariant.group !1
; CHECK-NEXT:   %cmp3.not233 = icmp eq i64 %cond, 0
; CHECK-NEXT:   br i1 %cmp3.not233, label %for.cond.cleanup4, label %for.cond6.preheader.preheader

; CHECK: for.cond6.preheader.preheader:                    ; preds = %_ZNK11OuterStruct4sizeEv.exit
; CHECK-NEXT:   %2 = getelementptr inbounds float*, float** %a17_malloccache, i64 %iv
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %cond, 4
; CHECK-NEXT:   %malloccall5 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %a17_malloccache6 = bitcast i8* %malloccall5 to float*
; CHECK-NEXT:   store float* %a17_malloccache6, float** %2, align 4, !invariant.group !2
; CHECK-NEXT:   %a17.pre = load float, float* %in, align 8
; CHECK-NEXT:   br label %for.cond6.preheader

; CHECK: for.cond6.preheader:                              ; preds = %for.cond6.preheader, %for.cond6.preheader.preheader
; CHECK-NEXT:   %iv3 = phi i64 [ %iv.next4, %for.cond6.preheader ], [ 0, %for.cond6.preheader.preheader ]
; CHECK-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:   %sq = fmul float %a17.pre, %a17.pre
; CHECK-NEXT:   store float %sq, float* %out, align 8
; CHECK-NEXT:   %3 = getelementptr inbounds float, float* %a17_malloccache6, i64 %iv3
; CHECK-NEXT:   store float %a17.pre, float* %3, align 4, !invariant.group !3
; CHECK-NEXT:   %cmp3.not = icmp eq i64 %iv.next4, %cond
; CHECK-NEXT:   br i1 %cmp3.not, label %for.cond.cleanup4, label %for.cond6.preheader

; CHECK: for.cond.cleanup4:                                ; preds = %for.cond6.preheader, %_ZNK11OuterStruct4sizeEv.exit
; CHECK-NEXT:   %cmp.not = icmp eq i64 %iv.next, 10
; CHECK-NEXT:   br i1 %cmp.not, label %for.cond.cleanup, label %for.body

; CHECK: for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
; CHECK-NEXT:   store float 0.000000e+00, float* %in, align 8
; CHECK-NEXT:   store float 0.000000e+00, float* %"in'", align 8
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: invertentry:                                      ; preds = %invertfor.body
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall9)
; CHECK-NEXT:   tail call void @free(i8* nonnull %malloccall13)
; CHECK-NEXT:   ret void

; CHECK: invertfor.body:                                   ; preds = %invertmerge
; CHECK-NEXT:   %4 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %4, label %invertentry, label %incinvertfor.body

; CHECK: incinvertfor.body:                                ; preds = %invertfor.body
; CHECK-NEXT:   %5 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond.cleanup4

; CHECK: incinvertfor.body2:                               ; preds = %invertmerge
; CHECK-NEXT:   %6 = add nsw i64 %"iv1'ac.0", -1
; CHECK-NEXT:   br label %invertmerge

; CHECK: invertmerge:                                      ; preds = %invert_ZNK11OuterStruct4sizeEv.exit, %incinvertfor.body2
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ 6, %invert_ZNK11OuterStruct4sizeEv.exit ], [ %6, %incinvertfor.body2 ]
; CHECK-NEXT:   %7 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %7, label %invertfor.body, label %incinvertfor.body2

; CHECK: invert_ZNK11OuterStruct4sizeEv.exit:              ; preds = %invertfor.cond.cleanup4, %invertfor.cond6.preheader.preheader
; CHECK-NEXT:   %"a17'de.0" = phi float [ %"a17'de.2", %invertfor.cond.cleanup4 ], [ 0.000000e+00, %invertfor.cond6.preheader.preheader ]
; CHECK-NEXT:   %"sq'de.0" = phi float [ %"sq'de.2", %invertfor.cond.cleanup4 ], [ 0.000000e+00, %invertfor.cond6.preheader.preheader ]
; CHECK-NEXT:   br label %invertmerge

; CHECK: invertfor.cond6.preheader.preheader:              ; preds = %invertfor.cond6.preheader
; CHECK-NEXT:   %8 = bitcast float* %.pre to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %8)
; CHECK-NEXT:   br label %invert_ZNK11OuterStruct4sizeEv.exit

; CHECK: invertfor.cond6.preheader:                        ; preds = %invertfor.cond.cleanup4.loopexit, %incinvertfor.cond6.preheader
; CHECK-NEXT:   %"a17'de.1" = phi float [ %"a17'de.2", %invertfor.cond.cleanup4.loopexit ], [ 0.000000e+00, %incinvertfor.cond6.preheader ]
; CHECK-NEXT:   %"sq'de.1" = phi float [ %"sq'de.2", %invertfor.cond.cleanup4.loopexit ], [ 0.000000e+00, %incinvertfor.cond6.preheader ]
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ %_unwrap12, %invertfor.cond.cleanup4.loopexit ], [ %18, %incinvertfor.cond6.preheader ]
; CHECK-NEXT:   %9 = load float, float* %"out'", align 8
; CHECK-NEXT:   store float 0.000000e+00, float* %"out'", align 8
; CHECK-NEXT:   %10 = fadd fast float %"sq'de.1", %9
; CHECK-NEXT:   %11 = getelementptr inbounds float, float* %.pre, i64 %"iv3'ac.0"
; CHECK-NEXT:   %12 = load float, float* %11, align 4, !invariant.group !3
; CHECK-NEXT:   %m0diffea17 = fmul fast float %10, %12
; CHECK-NEXT:   %13 = fadd fast float %"a17'de.1", %m0diffea17
; CHECK-NEXT:   %14 = fadd fast float %13, %m0diffea17
; CHECK-NEXT:   %15 = load float, float* %"in'", align 8
; CHECK-NEXT:   %16 = fadd fast float %15, %14
; CHECK-NEXT:   store float %16, float* %"in'", align 8
; CHECK-NEXT:   %17 = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   br i1 %17, label %invertfor.cond6.preheader.preheader, label %incinvertfor.cond6.preheader

; CHECK: incinvertfor.cond6.preheader:                     ; preds = %invertfor.cond6.preheader
; CHECK-NEXT:   %18 = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertfor.cond6.preheader

; CHECK: invertfor.cond.cleanup4.loopexit:                 ; preds = %invertfor.cond.cleanup4
; CHECK-NEXT:   %19 = getelementptr inbounds i64, i64* %cond.lcssa_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %20 = load i64, i64* %19, align 8, !invariant.group !0
; CHECK-NEXT:   %_unwrap12 = add i64 %20, -1
; CHECK-NEXT:   %.phi.trans.insert = getelementptr inbounds float*, float** %a17_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %.pre = load float*, float** %.phi.trans.insert, align 8, !invariant.group !2
; CHECK-NEXT:   br label %invertfor.cond6.preheader

; CHECK: invertfor.cond.cleanup4:                          ; preds = %for.cond.cleanup, %incinvertfor.body
; CHECK-NEXT:   %"a17'de.2" = phi float [ 0.000000e+00, %for.cond.cleanup ], [ %"a17'de.0", %incinvertfor.body ]
; CHECK-NEXT:   %"sq'de.2" = phi float [ 0.000000e+00, %for.cond.cleanup ], [ %"sq'de.0", %incinvertfor.body ]
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ 9, %for.cond.cleanup ], [ %5, %incinvertfor.body ]
; CHECK-NEXT:   %21 = getelementptr inbounds i64, i64* %s.0.lcssa.i_malloccache, i64 %"iv'ac.0"
; CHECK-NEXT:   %22 = load i64, i64* %21, align 8, !invariant.group !1
; CHECK-NEXT:   %cmp3.not233_unwrap = icmp eq i64 %22, 0
; CHECK-NEXT:   br i1 %cmp3.not233_unwrap, label %invert_ZNK11OuterStruct4sizeEv.exit, label %invertfor.cond.cleanup4.loopexit
; CHECK-NEXT: }
