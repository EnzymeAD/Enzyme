; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%57 = type { i32*, i32, float }

@enzyme_out = external dso_local global i32, align 4

define dso_local i32 @main() {
bb: 
  
  %i768 = alloca %57, align 8

  
  %i750 = alloca %57, align 8
  
  %i2661 = getelementptr inbounds %57, %57* %i750, i32 0, i32 1
  %i2671 = load i32, i32* @enzyme_out, align 4
  store i32 %i2671, i32* %i2661, align 8


  %i2672 = bitcast %57* %i750 to i8*
  %i2673 = getelementptr inbounds i8, i8* %i2672, i64 12
  %i2674 = bitcast i8* %i2673 to float*
  store float 0.0, float* %i2674, align 4
  
  %i2687 = bitcast %57* %i750 to { i32*, i64 }*
  %i2688 = load { i32*, i64 }, { i32*, i64 }* %i2687, align 8
  
  
  %i2692 = extractvalue { i32*, i64 } %i2688, 1


 
  %i2711 = bitcast %57* %i768 to { i32*, i64 }*
  %i2712 = getelementptr inbounds { i32*, i64 }, { i32*, i64 }* %i2711, i32 0, i32 1
  store i64 %i2692, i64* %i2712, align 8


  %i2717 = getelementptr inbounds %57, %57* %i768, i32 0, i32 2

  %i2719 = load float, float* %i2717, align 4

  %i2720 = call float (...) @_Z17__enzyme_autodiffIN6enzyme5tupleIJNS1_IJfEEEEEEJPvPiS5_ifS5_P8overloadEET_DpT0_(i8* bitcast (float (float)* @_ZN6enzyme6detail14templated_callI8overloadFffEE4wrapEfPS2_ to i8*), metadata !"enzyme_out", float %i2719)
  ret i32 0
}

define linkonce_odr dso_local float @_ZN6enzyme6detail14templated_callI8overloadFffEE4wrapEfPS2_(float %arg) {
bb:
  ret float %arg
}

declare dso_local float @_Z17__enzyme_autodiffIN6enzyme5tupleIJNS1_IJfEEEEEEJPvPiS5_ifS5_P8overloadEET_DpT0_(...)

; CHECK:   call { float } @diffe_ZN6enzyme6detail14templated_callI8overloadFffEE4wrapEfPS2_(float %i2719, float 1.000000e+00)
