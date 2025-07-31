; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

define internal fastcc void @d({ float*, float }* noalias nocapture nofree noundef nonnull writeonly sret({ float*, float }) %a0, float "enzyme_type"="{[-1]:Float@float}" %arrayref, float* "enzyme_type"="{[-1]:Pointer}" %a3) {
top:
  %a8 = fneg float %arrayref
  %a9 = insertvalue { float*, float } zeroinitializer, float* %a3, 0
  %a10 = insertvalue { float*, float } %a9, float %a8, 1
  store { float*, float } %a10, { float*, float }* %a0, align 8
  ret void
}

define float* @loss(float* %x, float %flt) {
top:
  %sret = alloca { float*, float }, align 8
  call fastcc void @d({ float*, float }* noalias nocapture nofree noundef nonnull writeonly sret({ float*, float }) align 8 dereferenceable(16) %sret, float %flt, float* %x)
  %getfield_addr = getelementptr inbounds { float*, float }, { float*, float }* %sret, i64 0, i32 0
  %getfield = load atomic float*, float** %getfield_addr unordered, align 8
  ret float* %getfield
}

declare i8* @__enzyme_virtualreverse(...) 

define i8* @dsquare(double %x) {
entry:
  %0 = call i8* (...) @__enzyme_virtualreverse(float* (float*, float)* nonnull @loss)
  ret i8* %0
}

; CHECK: define internal fastcc { float } @diffed({ float*, float }* noalias nocapture nofree writeonly "enzyme_sret" %a0, { float*, float }* nocapture nofree "enzyme_sret" %"a0'", float "enzyme_type"="{[-1]:Float@float}" %arrayref, float* "enzyme_type"="{[-1]:Pointer}" %a3, float* "enzyme_type"="{[-1]:Pointer}" %"a3'") #1 {
; CHECK-NEXT: top:
; CHECK-NEXT:   %0 = alloca { float*, float }, align 8
; CHECK-NEXT:   %"a10'de" = alloca { float*, float }, align 8
; CHECK-NEXT:   store { float*, float } zeroinitializer, { float*, float }* %"a10'de", align 8
; CHECK-NEXT:   %1 = load { float*, float }, { float*, float }* %"a0'", align 8
; CHECK-NEXT:   store { float*, float } zeroinitializer, { float*, float }* %0, align 8
; CHECK-NEXT:   %2 = bitcast { float*, float }* %"a0'" to i8*
; CHECK-NEXT:   %3 = getelementptr inbounds i8, i8* %2, i64 8
; CHECK-NEXT:   %4 = bitcast { float*, float }* %0 to i8*
; CHECK-NEXT:   %5 = getelementptr inbounds i8, i8* %4, i64 8
; CHECK-NEXT:   %6 = bitcast i8* %3 to i64*
; CHECK-NEXT:   %7 = bitcast i8* %5 to i64*
; CHECK-NEXT:   %8 = load i64, i64* %7, align 4
; CHECK-NEXT:   store i64 %8, i64* %6, align 8
; CHECK-NEXT:   %9 = extractvalue { float*, float } %1, 1
; CHECK-NEXT:   %10 = getelementptr inbounds { float*, float }, { float*, float }* %"a10'de", i32 0, i32 1
; CHECK-NEXT:   %11 = load float, float* %10, align 4
; CHECK-NEXT:   %12 = fadd fast float %11, %9
; CHECK-NEXT:   store float %12, float* %10, align 4
; CHECK-NEXT:   %13 = load { float*, float }, { float*, float }* %"a10'de", align 8
; CHECK-NEXT:   %14 = extractvalue { float*, float } %13, 1
; CHECK-NEXT:   store { float*, float } zeroinitializer, { float*, float }* %"a10'de", align 8
; CHECK-NEXT:   %15 = fneg fast float %14
; CHECK-NEXT:   %16 = insertvalue { float } undef, float %15, 0
; CHECK-NEXT:   ret { float } %16
; CHECK-NEXT: }
