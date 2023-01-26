; RUN: if [ %llvmver -lt 12 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -simplifycfg -early-cse -S | FileCheck %s ; fi


; #include <stdio.h>
; #include <array>

; using namespace std;

; extern int enzyme_width;

; struct Gradients {
;     array<double,3> dx1, dx2, dx3;
; };

; extern Gradients __enzyme_fwddiff(void*, ...);

; array<double,3> square(double x) {
;     return {x * x, x * x * x, x};
; }
; Gradients dsquare(double x) {
;     // This returns the derivative of square or 2 * x
;     return __enzyme_fwddiff((void*)square, enzyme_width, 3, x, 1.0, 2.0, 3.0);
; }
; int main() {
;     printf("%f \n", dsquare(3).dx1[0]);
; }


%"struct.std::array" = type { [3 x double] }
%"struct.std::array.vec" = type { [3 x <3 x double>] }

$_ZNSt5arrayIdLm3EEixEm = comdat any

$_ZNSt14__array_traitsIdLm3EE6_S_refERA3_Kdm = comdat any

@enzyme_width = external dso_local local_unnamed_addr global i32, align 4
@.str = private unnamed_addr constant [5 x i8] c"%f \0A\00", align 1

define dso_local void @_Z6squared(%"struct.std::array"* noalias nocapture sret align 8 %agg.result, double %x) #0 {
entry:
  %arrayinit.begin = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 0
  %mul = fmul double %x, %x
  store double %mul, double* %arrayinit.begin, align 8
  %arrayinit.element = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 1
  %mul2 = fmul double %mul, %x
  store double %mul2, double* %arrayinit.element, align 8
  %arrayinit.element3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 2
  store double %x, double* %arrayinit.element3, align 8
  ret void
}

define dso_local void @_Z7dsquared(%"struct.std::array.vec"* noalias sret align 8 %agg.result, double %x) local_unnamed_addr #1 {
entry:
  %0 = load i32, i32* @enzyme_width, align 4
  call void (%"struct.std::array.vec"*, i8*, ...) @_Z16__enzyme_fwddiffPvz(%"struct.std::array.vec"* sret align 8 %agg.result, i8* bitcast (void (%"struct.std::array"*, double)* @_Z6squared to i8*), i32 %0, i32 3, double %x, <3 x double> <double 1.000000e+00, double 2.000000e+00, double 3.000000e+00>)
  ret void
}

declare dso_local void @_Z16__enzyme_fwddiffPvz(%"struct.std::array.vec"* sret align 8, i8*, ...) local_unnamed_addr #2

define dso_local i32 @main() local_unnamed_addr #3 {
entry:
  %ref.tmp = alloca %"struct.std::array.vec", align 8
  %0 = bitcast %"struct.std::array.vec"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 72, i8* nonnull %0) #7
  call void @_Z7dsquared(%"struct.std::array.vec"* nonnull sret align 8 %ref.tmp, double 3.000000e+00)
  call void @llvm.lifetime.end.p0i8(i64 72, i8* nonnull %0) #7
  ret i32 0
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #5

define linkonce_odr dso_local nonnull align 8 dereferenceable(8) double* @_ZNSt5arrayIdLm3EEixEm(%"struct.std::array"* %this, i64 %__n) local_unnamed_addr #6 comdat align 2 {
entry:
  %_M_elems = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %this, i64 0, i32 0
  %call = call nonnull align 8 dereferenceable(8) double* @_ZNSt14__array_traitsIdLm3EE6_S_refERA3_Kdm([3 x double]* nonnull align 8 dereferenceable(24) %_M_elems, i64 %__n) #7
  ret double* %call
}

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #5

define linkonce_odr dso_local nonnull align 8 dereferenceable(8) double* @_ZNSt14__array_traitsIdLm3EE6_S_refERA3_Kdm([3 x double]* nonnull align 8 dereferenceable(24) %__t, i64 %__n) local_unnamed_addr #6 comdat align 2 {
entry:
  %arrayidx = getelementptr inbounds [3 x double], [3 x double]* %__t, i64 0, i64 %__n
  ret double* %arrayidx
}

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { nounwind }


; CHECK: define internal void @fwddiffe3_Z6squared(%"struct.std::array"* noalias nocapture align 8 "enzyme_sret" %agg.result, %"struct.std::array.vec"* "enzyme_sret_v" %"agg.result'", double %x, <3 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"arrayinit.begin'ipg" = getelementptr inbounds %"struct.std::array.vec", %"struct.std::array.vec"* %"agg.result'", i64 0, i32 0, i64 0
; CHECK-NEXT:   %arrayinit.begin = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 0
; CHECK-NEXT:   %mul = fmul double %x, %x
; CHECK-NEXT:   %.splatinsert = insertelement <3 x double> undef, double %x, i32 0
; CHECK-NEXT:   %.splat = shufflevector <3 x double> %.splatinsert, <3 x double> undef, <3 x i32> zeroinitializer
; CHECK-NEXT:   %0 = fmul fast <3 x double> %"x'", %.splat
; CHECK-NEXT:   %1 = fadd fast <3 x double> %0, %0
; CHECK-NEXT:   store double %mul, double* %arrayinit.begin, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   store <3 x double> %1, <3 x double>* %"arrayinit.begin'ipg", align 8, !alias.scope !7, !noalias !8
; CHECK-NEXT:   %"arrayinit.element'ipg" = getelementptr inbounds %"struct.std::array.vec", %"struct.std::array.vec"* %"agg.result'", i64 0, i32 0, i64 1
; CHECK-NEXT:   %arrayinit.element = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 1
; CHECK-NEXT:   %mul2 = fmul double %mul, %x
; CHECK-NEXT:   %.splatinsert3 = insertelement <3 x double> undef, double %mul, i32 0
; CHECK-NEXT:   %.splat4 = shufflevector <3 x double> %.splatinsert3, <3 x double> undef, <3 x i32> zeroinitializer
; CHECK-NEXT:   %2 = fmul fast <3 x double> %1, %.splat
; CHECK-NEXT:   %3 = fmul fast <3 x double> %"x'", %.splat4
; CHECK-NEXT:   %4 = fadd fast <3 x double> %2, %3
; CHECK-NEXT:   store double %mul2, double* %arrayinit.element, align 8, !alias.scope !9, !noalias !12
; CHECK-NEXT:   store <3 x double> %4, <3 x double>* %"arrayinit.element'ipg", align 8, !alias.scope !16, !noalias !17
; CHECK-NEXT:   %"arrayinit.element3'ipg" = getelementptr inbounds %"struct.std::array.vec", %"struct.std::array.vec"* %"agg.result'", i64 0, i32 0, i64 2
; CHECK-NEXT:   %arrayinit.element3 = getelementptr inbounds %"struct.std::array", %"struct.std::array"* %agg.result, i64 0, i32 0, i64 2
; CHECK-NEXT:   store double %x, double* %arrayinit.element3, align 8, !alias.scope !18, !noalias !21
; CHECK-NEXT:   store <3 x double> %"x'", <3 x double>* %"arrayinit.element3'ipg", align 8, !alias.scope !25, !noalias !26
; CHECK-NEXT:   ret void
; CHECK-NEXT: }