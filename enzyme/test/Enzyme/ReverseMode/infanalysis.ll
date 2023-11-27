; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

@_ZTV4Test = linkonce_odr dso_local unnamed_addr constant [1 x i8*] [i8* bitcast ({ i8**, i8* }* @_ZTI4Test to i8*)], align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTI4Test = linkonce_odr dso_local constant { i8**, i8* } { i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i8* null }, align 8

define void @_Z8simulatev() {
entry:
  %sys = alloca i8**, align 8
  store i8** getelementptr inbounds ([1 x i8*], [1 x i8*]* @_ZTV4Test, i32 0, i32 1), i8*** %sys, align 8
  ret void
}

define void @main() {
entry:
  %call = call double @_Z17__enzyme_autodiffPv(void ()* @_Z8simulatev)
  ret void
}

declare double @_Z17__enzyme_autodiffPv(void ()*)

; CHECK: define internal void @diffe_Z8simulatev
