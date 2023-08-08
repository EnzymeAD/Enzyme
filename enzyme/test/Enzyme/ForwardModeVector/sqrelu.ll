; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -early-cse -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse,adce)" -enzyme-preopt=false -S | FileCheck %s
;
%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: nounwind readnone uwtable
define dso_local double @sqrelu(double %x) #0 {
entry:
  %cmp = fcmp fast ogt double %x, 0.000000e+00
  br i1 %cmp, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %0 = tail call fast double @llvm.sin.f64(double %x)
  %mul = fmul fast double %0, %x
  %1 = tail call fast double @llvm.sqrt.f64(double %mul)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi double [ %1, %cond.true ], [ 0.000000e+00, %entry ]
  ret double %cond
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double) #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sqrt.f64(double) #1

; Function Attrs: nounwind uwtable
define dso_local %struct.Gradients @dsqrelu(double %x) local_unnamed_addr #2 {
entry:
  %0 = tail call %struct.Gradients (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @sqrelu, metadata !"enzyme_width", i64 2, double %x, double 1.0, double 1.5)
  ret %struct.Gradients %0
}

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind }

; CHECK: define internal [2 x double] @fwddiffe2sqrelu(double [[X:%.*]], [2 x double] %"x'")
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = fcmp fast ogt double [[X]], 0.000000e+00
; CHECK-NEXT:    br i1 [[CMP]], label [[COND_TRUE:%.*]], label [[COND_END:%.*]]
; CHECK:       cond.true:
; CHECK-NEXT:    [[TMP0:%.*]] = tail call fast double @llvm.sin.f64(double [[X]])
; CHECK-NEXT:    [[TMP1:%.*]] = call fast double @llvm.cos.f64(double [[X]])
; CHECK-NEXT:    [[TMP2:%.*]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:    [[TMP3:%.*]] = fmul fast double [[TMP2]], [[TMP1]]
; CHECK-NEXT:    [[TMP5:%.*]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:    [[TMP6:%.*]] = fmul fast double [[TMP5]], [[TMP1]]
; CHECK-NEXT:    [[MUL:%.*]] = fmul fast double [[TMP0]], [[X]]
; CHECK-NEXT:    [[TMP7:%.*]] = fmul fast double [[TMP3]], [[X]]
; CHECK-NEXT:    [[TMP11:%.*]] = fmul fast double [[TMP6]], [[X]]
; CHECK-NEXT:    [[TMP8:%.*]] = fmul fast double [[TMP2]], [[TMP0]]
; CHECK-NEXT:    [[TMP12:%.*]] = fmul fast double [[TMP5]], [[TMP0]]
; CHECK-NEXT:    [[TMP9:%.*]] = fadd fast double [[TMP7]], [[TMP8]]
; CHECK-NEXT:    [[TMP13:%.*]] = fadd fast double [[TMP11]], [[TMP12]]
; CHECK-NEXT:    [[TMP17:%.*]] = fcmp fast ueq double [[MUL]], 0.000000e+00
; CHECK-NEXT:    [[TMP14:%.*]] = call fast double @llvm.sqrt.f64(double [[MUL]])
; CHECK-NEXT:    [[TMP15:%.*]] = fmul fast double 2.000000e+00, [[TMP14]]
; CHECK-NEXT:    [[TMP16:%.*]] = fdiv fast double [[TMP9]], [[TMP15]]
; CHECK-NEXT:    [[TMP21:%.*]] = fdiv fast double [[TMP13]], [[TMP15]]
; CHECK-NEXT:    [[TMP18:%.*]] = select {{(fast )?}}i1 [[TMP17]], double 0.000000e+00, double [[TMP16]]
; CHECK-NEXT:    [[TMP22:%.*]] = select {{(fast )?}}i1 [[TMP17]], double 0.000000e+00, double [[TMP21]]
; CHECK-NEXT:    br label [[COND_END]]
; CHECK:       cond.end:
; CHECK-NEXT:    [[TMP23:%.*]] = phi {{(fast )?}}double [ [[TMP18]], [[COND_TRUE]] ], [ 0.000000e+00, [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[TMP24:%.*]] = phi {{(fast )?}}double [ [[TMP22]], [[COND_TRUE]] ], [ 0.000000e+00, [[ENTRY]] ]
; CHECK-NEXT:    [[TMP25:%.*]] = insertvalue [2 x double] undef, double [[TMP23]], 0
; CHECK-NEXT:    [[TMP26:%.*]] = insertvalue [2 x double] [[TMP25]], double [[TMP24]], 1
; CHECK-NEXT:    ret [2 x double] [[TMP26]]
