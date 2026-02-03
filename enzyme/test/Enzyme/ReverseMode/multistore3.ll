; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg)" -S | FileCheck %s

declare i8* @__enzyme_virtualreverse(...)

define void @square({ {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* sret({ {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }) "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Integer, [-1,32]:Float@double, [-1,40]:Pointer}" %out, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %in) {
entry:
  store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %in, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %out
  ret void
}

define i8* @dsquare(double %x) {
entry:
  %0 = tail call i8* (...) @__enzyme_virtualreverse(void ({ {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }*, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* })* nonnull @square)
  ret i8* %0
}

; CHECK: define internal i8* @augmented_square({ {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* nocapture writeonly "enzyme_sret"="{{[0-9]+}}" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Integer, [-1,32]:Float@double, [-1,40]:Pointer}" %out, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* nocapture "enzyme_sret"="{{[0-9]+}}" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Integer, [-1,32]:Float@double, [-1,40]:Pointer}" %"out'", { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %in, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %"in'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }
; CHECK-NEXT:   %1 = alloca { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }
; CHECK-NEXT:   %2 = alloca { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }
; CHECK-NEXT:   store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %"in'", { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %0
; CHECK-NEXT:   %3 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"out'" to i64*
; CHECK-NEXT:   %4 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %0 to i64*
; CHECK-NEXT:   %5 = load i64, i64* %4
; CHECK-NEXT:   store i64 %5, i64* %3
; CHECK-NEXT:   store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %"in'", { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %1
; CHECK-NEXT:   %6 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"out'" to i8*
; CHECK-NEXT:   %7 = getelementptr inbounds i8, i8* %6, i64 24
; CHECK-NEXT:   %8 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %1 to i8*
; CHECK-NEXT:   %9 = getelementptr inbounds i8, i8* %8, i64 24
; CHECK-NEXT:   %10 = bitcast i8* %7 to i64*
; CHECK-NEXT:   %11 = bitcast i8* %9 to i64*
; CHECK-NEXT:   %12 = load i64, i64* %11
; CHECK-NEXT:   store i64 %12, i64* %10
; CHECK-NEXT:   store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %"in'", { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %2
; CHECK-NEXT:   %13 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"out'" to i8*
; CHECK-NEXT:   %14 = getelementptr inbounds i8, i8* %13, i64 40
; CHECK-NEXT:   %15 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %2 to i8*
; CHECK-NEXT:   %16 = getelementptr inbounds i8, i8* %15, i64 40
; CHECK-NEXT:   %17 = bitcast i8* %14 to i64*
; CHECK-NEXT:   %18 = bitcast i8* %16 to i64*
; CHECK-NEXT:   %19 = load i64, i64* %18
; CHECK-NEXT:   store i64 %19, i64* %17
; CHECK-NEXT:   store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %in, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %out
; CHECK-NEXT:   ret i8* null
; CHECK-NEXT: }

; CHECK: define internal void @diffesquare({ {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* nocapture writeonly "enzyme_sret"="{{[0-9]+}}" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Integer, [-1,32]:Float@double, [-1,40]:Pointer}" %out, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* nocapture "enzyme_sret"="{{[0-9]+}}" "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Integer, [-1,32]:Float@double, [-1,40]:Pointer}" %"out'", { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %in, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %"in'", i8* %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @free(i8* nonnull %tapeArg)
; CHECK-NEXT:   %0 = alloca { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }
; CHECK-NEXT:   %"in'de" = alloca { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }
; CHECK-NEXT:   store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } zeroinitializer, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"in'de"
; CHECK-NEXT:   %1 = alloca { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }
; CHECK-NEXT:   %2 = load { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"out'"
; CHECK-NEXT:   store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } zeroinitializer, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %0
; CHECK-NEXT:   %3 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"out'" to i8*
; CHECK-NEXT:   %4 = getelementptr inbounds i8, i8* %3, i64 8
; CHECK-NEXT:   %5 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %0 to i8*
; CHECK-NEXT:   %6 = getelementptr inbounds i8, i8* %5, i64 8
; CHECK-NEXT:   %7 = bitcast i8* %4 to [2 x i64]*
; CHECK-NEXT:   %8 = bitcast i8* %6 to [2 x i64]*
; CHECK-NEXT:   %9 = load [2 x i64], [2 x i64]* %8, align 4
; CHECK-NEXT:   store [2 x i64] %9, [2 x i64]* %7, align 8, !alias.scope !5, !noalias !8
; CHECK-NEXT:   %10 = extractvalue { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %2, 1
; CHECK-NEXT:   %11 = getelementptr inbounds { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"in'de", i32 0, i32 1
; CHECK-NEXT:   %12 = load double, double* %11, align 8
; CHECK-NEXT:   %13 = fadd fast double %12, %10
; CHECK-NEXT:   store double %13, double* %11, align 8
; CHECK-NEXT:   %14 = extractvalue { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %2, 2
; CHECK-NEXT:   %15 = getelementptr inbounds { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"in'de", i32 0, i32 2
; CHECK-NEXT:   %16 = load double, double* %15
; CHECK-NEXT:   %17 = fadd fast double %16, %14
; CHECK-NEXT:   store double %17, double* %15
; CHECK-NEXT:   %18 = load { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"out'"
; CHECK-NEXT:   store { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } zeroinitializer, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %1
; CHECK-NEXT:   %19 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"out'" to i8*
; CHECK-NEXT:   %20 = getelementptr inbounds i8, i8* %19, i64 32
; CHECK-NEXT:   %21 = bitcast { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %1 to i8*
; CHECK-NEXT:   %22 = getelementptr inbounds i8, i8* %21, i64 32
; CHECK-NEXT:   %23 = bitcast i8* %20 to i64*
; CHECK-NEXT:   %24 = bitcast i8* %22 to i64*
; CHECK-NEXT:   %25 = load i64, i64* %24
; CHECK-NEXT:   store i64 %25, i64* %23
; CHECK-NEXT:   %26 = extractvalue { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* } %18, 4
; CHECK-NEXT:   %27 = getelementptr inbounds { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }, { {} addrspace(10)*, double, double, i8, double, {} addrspace(10)* }* %"in'de", i32 0, i32 4
; CHECK-NEXT:   %28 = load double, double* %27
; CHECK-NEXT:   %29 = fadd fast double %28, %26
; CHECK-NEXT:   store double %29, double* %27
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
