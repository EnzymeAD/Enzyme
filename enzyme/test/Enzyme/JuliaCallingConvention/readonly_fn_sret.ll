; RUN: if [ %llvmver -ge 15 ]; then %opt %OPnewLoadEnzyme -S -passes=enzyme-fixup-julia < %s | FileCheck %s; fi

; A reverse-pass callee that only reads the shadow it receives through an
; enzyme_sret argument is a readonly function. Once the fixup returns that
; shadow together with the result through an sret, the function writes
; argument memory, so neither it nor its call site may stay readonly: LLVM
; would otherwise forward the caller's zero-initialization of the sret buffer
; past the call and read back a zero result (EnzymeAD/Enzyme.jl#3679).
; Likewise enzyme_ReadOnlyOrThrow, which rules out writes visible to the
; caller, becomes its local variant, which allows writing an sret.

; CHECK: define double @caller(double %arg, ptr %sret_box)
; CHECK:   call fastcc void @readonly_fn(ptr sret({ double, { [2 x double], double, i64 } }) %stack_sret, double %arg) #[[CALLATTR:[0-9]+]]
; CHECK: define internal fastcc void @readonly_fn(ptr noalias sret({ double, { [2 x double], double, i64 } }) %0, double %1) #[[FNATTR:[0-9]+]]
; CHECK: attributes #[[FNATTR]] = { {{(argmemonly nofree nosync nounwind willreturn|nofree nosync nounwind willreturn memory\(argmem: readwrite\))}} "enzyme_LocalReadOnlyOrThrow" }

define double @caller(double %arg, ptr %sret_box) {
entry:
  %res = call fastcc double @readonly_fn(ptr readonly "enzyme_sret"="test_type4" %sret_box, double %arg) #0
  ret double %res
}

define internal fastcc double @readonly_fn(ptr noalias nocapture nofree noundef nonnull readonly align 8 dereferenceable(32) "enzyme_sret"="test_type4" %0, double %1) #0 {
top:
  %gep = getelementptr inbounds { [2 x double], double, i64 }, ptr %0, i32 0, i32 1
  %val = load double, ptr %gep, align 8
  %r = fdiv double %val, %1
  ret double %r
}

attributes #0 = { argmemonly nofree nosync nounwind readonly willreturn "enzyme_ReadOnlyOrThrow" }
