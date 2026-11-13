; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%"struct.Eigen::internal::CacheSizes" = type { i64, i64, i64 }

@_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes = linkonce_odr global %"struct.Eigen::internal::CacheSizes" zeroinitializer, align 8

declare void @_ZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_(i32, i64*, i64*, i64*)

define double @tester(double %x, i64* %l1, i64* %l2, i64* %l3) {
entry:
  call void @_ZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_(i32 0, i64* %l1, i64* %l2, i64* %l3)
  %gep = getelementptr inbounds %"struct.Eigen::internal::CacheSizes", %"struct.Eigen::internal::CacheSizes"* @_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes, i32 0, i32 0
  %val = load i64, i64* %gep, align 8
  store i64 %val, i64* %l1, align 8
  %sq = fmul double %x, %x
  ret double %sq
}

define double @test_derivative(double %x, double %dx, i64* %l1, i64* %l2, i64* %l3) {
entry:
  %0 = tail call double (double (double, i64*, i64*, i64*)*, ...) @__enzyme_fwddiff(double (double, i64*, i64*, i64*)* nonnull @tester, double %x, double %dx, metadata !"enzyme_const", i64* %l1, metadata !"enzyme_const", i64* %l2, metadata !"enzyme_const", i64* %l3)
  ret double %0
}

declare double @__enzyme_fwddiff(double (double, i64*, i64*, i64*)*, ...)

; CHECK: define internal double @fwddiffetester(double %x, double %"x'", {{(ptr|i64\*)}} %l1, {{(ptr|i64\*)}} %l2, {{(ptr|i64\*)}} %l3)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @_ZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_(i32 0, {{(ptr|i64\*)}} %l1, {{(ptr|i64\*)}} %l2, {{(ptr|i64\*)}} %l3)
; CHECK-NEXT:   %val = load i64, {{.*}}@_ZZN5Eigen8internal20manage_caching_sizesENS_6ActionEPlS2_S2_E12m_cacheSizes{{.*}}
; CHECK-NEXT:   store i64 %val, {{(ptr|i64\*)}} %l1, align 8
; CHECK-NEXT:   %[[M0:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[M1:.+]] = fmul fast double %"x'", %x
; CHECK-NEXT:   %[[ADD:.+]] = fadd fast double %[[M0]], %[[M1]]
; CHECK-NEXT:   ret double %[[ADD]]
; CHECK-NEXT: }
