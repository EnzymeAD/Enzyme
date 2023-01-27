; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

%struct.RT = type { i64, [10 x [20 x double]] } ; 8 + 1600
%struct.ST = type { i64, double, %struct.RT } ; 8 + 8 + 1608

define double @tester(double %d) {
entry:
  %s = alloca %struct.ST
  store %struct.ST zeroinitializer, %struct.ST* %s
  %idx = getelementptr inbounds %struct.ST, %struct.ST* %s, i64 0, i32 2, i32 1, i64 5, i64 13
  store double %d, double* %idx
  %ptr = bitcast %struct.ST* %s to i8*
  %arrayidx = getelementptr inbounds i8, i8* %ptr, i64 768
  %cast = bitcast i8* %arrayidx to double*
  %res = load double, double* %cast
  ret double %res
}

define [2 x double] @test_derivative(double %x) {
entry:
  %call = call [2 x double] (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, [2 x double] [double 1.0, double 0.0])
  ret [2 x double] %call
}

declare [2 x double] @__enzyme_fwddiff(double (double)*, ...)
