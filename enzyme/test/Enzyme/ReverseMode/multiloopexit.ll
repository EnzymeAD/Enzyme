; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

declare i1 @end()

define float @todiff(float %a0, i64 %a1) {
entry:
  br label %loop1

loop1:                                             ; preds = %L19.i, %entry
  %sum.0 = phi float [ 0.000000e+00, %entry ], [ %sum.1, %floop1 ]
  %sum.1 = fadd float %sum.0, %a0
  %end1 = call i1 @end()
  br i1 %end1, label %loop2, label %floop1

loop2:                                             ; preds = %L2.i, %pass.i
  %sum.2 = phi float [ %sum.3, %pass.i ], [ %sum.1, %loop1 ]
  %end2 = call i1 @end()
  br i1 %end2, label %exit, label %pass.i

pass.i:                                           ; preds = %L9.i
  %sum.3 = fadd float %sum.2, %a0
  %end3 = call i1 @end()
  br i1 %end3, label %loop2, label %floop1

floop1:                                            ; preds = %pass.i, %L2.i
  br label %loop1

exit:                           ; preds = %L9.i
  ret float %sum.2
}

declare float @__enzyme_autodiff(...) 

define float @c() {
  %c = call float (...) @__enzyme_autodiff(float (float, i64)* @todiff, float 1.0, i64 1)
  ret float %c
}

; CHECK: define internal { float } @diffetodiff(float %a0, i64 %a1, float %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"iv'ac" = alloca i64, align 8
; CHECK-NEXT:   %loopLimit_cache = alloca i64, align 8
; CHECK-NEXT:   %"iv1'ac" = alloca i64, align 8
; CHECK-NEXT:   %loopLimit_cache2 = alloca i64*, align 8
; CHECK-NEXT:   %"sum.2'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.2'de", align 4
; CHECK-NEXT:   %"a0'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"a0'de", align 4
; CHECK-NEXT:   %"sum.1'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.1'de", align 4
; CHECK-NEXT:   %"sum.0'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.0'de", align 4
; CHECK-NEXT:   %"sum.3'de" = alloca float, align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.3'de", align 4
; CHECK-NEXT:   %end1_cache = alloca i1*, align 8
; CHECK-NEXT:   store i64* null, i64** %loopLimit_cache2, align 8
; CHECK-NEXT:   store i1* null, i1** %end1_cache, align 8
; CHECK-NEXT:   br label %loop1

; CHECK: loop1:                                            ; preds = %floop1, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %floop1 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1

; CHECK:   %end1 = call i1 @end()
; CHECK-NEXT:   %[[i32:.+]] = load i1*, i1** %end1_cache, align 8
; CHECK-NEXT:   %[[i33:.+]] = getelementptr inbounds i1, i1* %[[i32]], i64 %iv
; CHECK-NEXT:   store i1 %end1, i1* %[[i33]], align 1
; CHECK-NEXT:   br i1 %end1, label %loop2.preheader, label %floop1

; CHECK: loop2.preheader: 
; CHECK-NEXT:   br label %loop2

; CHECK: loop2:  
; CHECK-NEXT:   %iv1 = phi i64 [ 0, %loop2.preheader ], [ %iv.next2, %pass.i ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %end2 = call i1 @end()
; CHECK-NEXT:   br i1 %end2, label %exit, label %pass.i

; CHECK: pass.i:                                           ; preds = %loop2
; CHECK-NEXT:   %end3 = call i1 @end()
; CHECK-NEXT:   br i1 %end3, label %loop2, label %floop1.loopexit

; CHECK: floop1.loopexit:    
; CHECK-NEXT:   %[[i34:.+]] = phi i64 [ %iv1, %pass.i ]
; CHECK-NEXT:   %[[i35:.+]] = load i64*, i64** %loopLimit_cache2, align 8
; CHECK-NEXT:   %[[i36:.+]] = getelementptr inbounds i64, i64* %[[i35]], i64 %iv
; CHECK-NEXT:   store i64 %[[i34]], i64* %[[i36]], align 8
; CHECK-NEXT:   br label %floop1

; CHECK: floop1:                                           ; preds = %floop1.loopexit, %__enzyme_exponentialallocation.exit15
; CHECK-NEXT:   br label %loop1

; CHECK: exit:                                             ; preds = %loop2
; CHECK-NEXT:   %[[i37:.+]] = phi i64 [ %iv1, %loop2 ]
; CHECK-NEXT:   %[[i38:.+]] = phi i64 [ %iv, %loop2 ]
; CHECK-NEXT:   %[[i39:.+]] = load i64*, i64** %loopLimit_cache2, align 8
; CHECK-NEXT:   %[[i40:.+]] = getelementptr inbounds i64, i64* %[[i39]], i64 %iv
; CHECK-NEXT:   store i64 %[[i37]], i64* %[[i40]], align 8
; CHECK-NEXT:   store i64 %[[i38]], i64* %loopLimit_cache, align 8
; CHECK-NEXT:   br label %invertexit

; CHECK: invertentry:                                      ; preds = %invertloop1
; CHECK-NEXT:   %[[i41:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %forfree = load i64*, i64** %loopLimit_cache2, align 8
; CHECK-NEXT:   %[[i42:.+]] = bitcast i64* %forfree to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i42]])
; CHECK-NEXT:   %[[i43:.+]] = load float, float* %"a0'de", align 4
; CHECK-NEXT:   %[[i44:.+]] = insertvalue { float } undef, float %[[i43]], 0
; CHECK-NEXT:   %[[i45:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %forfree10 = load i1*, i1** %end1_cache, align 8
; CHECK-NEXT:   %[[i46:.+]] = bitcast i1* %forfree10 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i46]])
; CHECK-NEXT:   ret { float } %[[i44]]

; CHECK: invertloop1:                                      ; preds = %invertfloop1, %invertloop2.preheader
; CHECK-NEXT:   %[[i47:.+]] = load float, float* %"sum.1'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.1'de", align 4
; CHECK-NEXT:   %[[i48:.+]] = load float, float* %"sum.0'de", align 4
; CHECK-NEXT:   %[[i49:.+]] = fadd fast float %[[i48]], %[[i47]]
; CHECK-NEXT:   store float %[[i49]], float* %"sum.0'de", align 4
; CHECK-NEXT:   %[[i50:.+]] = load float, float* %"a0'de", align 4
; CHECK-NEXT:   %[[i51:.+]] = fadd fast float %[[i50]], %[[i47]]
; CHECK-NEXT:   store float %[[i51]], float* %"a0'de", align 4
; CHECK-NEXT:   %[[i52:.+]] = load float, float* %"sum.0'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.0'de", align 4
; CHECK-NEXT:   %[[i53:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[i54:.+]] = icmp eq i64 %[[i53]], 0
; CHECK-NEXT:   %[[i55:.+]] = xor i1 %[[i54]], true
; CHECK-NEXT:   %[[i56:.+]] = select fast i1 %[[i55]], float %[[i52]], float 0.000000e+00
; CHECK-NEXT:   %[[i57:.+]] = load float, float* %"sum.1'de", align 4
; CHECK-NEXT:   %[[i58:.+]] = fadd fast float %[[i57]], %[[i52]]
; CHECK-NEXT:   %[[i59:.+]] = select fast i1 %[[i54]], float %[[i57]], float %[[i58]]
; CHECK-NEXT:   store float %[[i59]], float* %"sum.1'de", align 4
; CHECK-NEXT:   br i1 %[[i54]], label %invertentry, label %incinvertloop1

; CHECK: incinvertloop1:                                   ; preds = %invertloop1
; CHECK-NEXT:   %[[i60:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[i61:.+]] = add nsw i64 %[[i60]], -1
; CHECK-NEXT:   store i64 %[[i61]], i64* %"iv'ac", align 4
; CHECK-NEXT:   br label %invertfloop1

; CHECK: invertloop2.preheader:                            ; preds = %invertloop2
; CHECK-NEXT:   br label %invertloop1

; CHECK: invertloop2:                                      ; preds = %mergeinvertloop2_exit, %invertpass.i
; CHECK-NEXT:   %[[i62:.+]] = load float, float* %"sum.2'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.2'de", align 4
; CHECK-NEXT:   %[[i63:.+]] = load i64, i64* %"iv1'ac", align 4
; CHECK-NEXT:   %[[i64:.+]] = icmp eq i64 %[[i63]], 0
; CHECK-NEXT:   %[[i65:.+]] = xor i1 %[[i64]], true
; CHECK-NEXT:   %[[i66:.+]] = select fast i1 %[[i64]], float %[[i62]], float 0.000000e+00
; CHECK-NEXT:   %[[i67:.+]] = load float, float* %"sum.1'de", align 4
; CHECK-NEXT:   %[[i68:.+]] = fadd fast float %[[i67]], %[[i62]]
; CHECK-NEXT:   %[[i69:.+]] = select fast i1 %[[i64]], float %[[i68]], float %[[i67]]
; CHECK-NEXT:   store float %[[i69]], float* %"sum.1'de", align 4
; CHECK-NEXT:   %[[i70:.+]] = select fast i1 %[[i65]], float %[[i62]], float 0.000000e+00
; CHECK-NEXT:   %[[i71:.+]] = load float, float* %"sum.3'de", align 4
; CHECK-NEXT:   %[[i72:.+]] = fadd fast float %[[i71]], %[[i62]]
; CHECK-NEXT:   %[[i73:.+]] = select fast i1 %[[i64]], float %[[i71]], float %[[i72]]
; CHECK-NEXT:   store float %[[i73]], float* %"sum.3'de", align 4
; CHECK-NEXT:   br i1 %[[i64]], label %invertloop2.preheader, label %incinvertloop2

; CHECK: incinvertloop2:                                   ; preds = %invertloop2
; CHECK-NEXT:   %[[i74:.+]] = load i64, i64* %"iv1'ac", align 4
; CHECK-NEXT:   %[[i75:.+]] = add nsw i64 %[[i74]], -1
; CHECK-NEXT:   store i64 %[[i75]], i64* %"iv1'ac", align 4
; CHECK-NEXT:   br label %invertpass.i

; CHECK: invertpass.i:                                     ; preds = %mergeinvertloop2_floop1.loopexit, %incinvertloop2
; CHECK-NEXT:   %[[i76:.+]] = load float, float* %"sum.3'de", align 4
; CHECK-NEXT:   store float 0.000000e+00, float* %"sum.3'de", align 4
; CHECK-NEXT:   %[[i77:.+]] = load float, float* %"sum.2'de", align 4
; CHECK-NEXT:   %[[i78:.+]] = fadd fast float %[[i77]], %[[i76]]
; CHECK-NEXT:   store float %[[i78]], float* %"sum.2'de", align 4
; CHECK-NEXT:   %[[i79:.+]] = load float, float* %"a0'de", align 4
; CHECK-NEXT:   %[[i80:.+]] = fadd fast float %[[i79]], %[[i76]]
; CHECK-NEXT:   store float %[[i80]], float* %"a0'de", align 4
; CHECK-NEXT:   br label %invertloop2

; CHECK: invertfloop1.loopexit:                            ; preds = %invertfloop1
; CHECK-NEXT:   %[[i81:.+]] = load i64*, i64** %loopLimit_cache2, align 8
; CHECK-NEXT:   %[[i82:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[i83:.+]] = getelementptr inbounds i64, i64* %[[i81]], i64 %[[i82]]
; CHECK-NEXT:   %[[i84:.+]] = load i64, i64* %[[i83]], align 8, !invariant.group !
; CHECK-NEXT:   br label %mergeinvertloop2_floop1.loopexit

; CHECK: mergeinvertloop2_floop1.loopexit:                 ; preds = %invertfloop1.loopexit
; CHECK-NEXT:   store i64 %[[i84]], i64* %"iv1'ac", align 4
; CHECK-NEXT:   br label %invertpass.i

; CHECK: invertfloop1:                                     ; preds = %incinvertloop1
; CHECK-NEXT:   %[[i85:.+]] = load i64, i64* %"iv'ac", align 4
; CHECK-NEXT:   %[[i86:.+]] = load i1*, i1** %end1_cache, align 8
; CHECK-NEXT:   %[[i87:.+]] = getelementptr inbounds i1, i1* %[[i86]], i64 %[[i85]]
; CHECK-NEXT:   %[[i88:.+]] = load i1, i1* %[[i87]], align 1, !invariant.group !
; CHECK-NEXT:   br i1 %[[i88]], label %invertfloop1.loopexit, label %invertloop1

; CHECK: invertexit:                                       ; preds = %exit
; CHECK-NEXT:   store float %differeturn, float* %"sum.2'de", align 4
; CHECK-NEXT:   %[[i89:.+]] = load i64, i64* %loopLimit_cache, align 8
; CHECK-NEXT:   %[[i90:.+]] = load i64*, i64** %loopLimit_cache2, align 8
; CHECK-NEXT:   %[[i92:.+]] = getelementptr inbounds i64, i64* %[[i90]], i64 %[[i89]]
; CHECK-NEXT:   %[[i93:.+]] = load i64, i64* %[[i92]], align 8
; CHECK-NEXT:   br label %mergeinvertloop2_exit

; CHECK: mergeinvertloop2_exit:                            ; preds = %invertexit
; CHECK-NEXT:   store i64 %[[i89]], i64* %"iv'ac", align 4
; CHECK-NEXT:   store i64 %[[i93]], i64* %"iv1'ac", align 4
; CHECK-NEXT:   br label %invertloop2
; CHECK-NEXT: }
