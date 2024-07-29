; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S -o /dev/null | FileCheck %s

declare void @__enzyme_autodiff(void (ptr, ptr, i64, i1)*, ...) 

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

define void @cpy(i1 %c, ptr %0, ptr %1) {
entry:
  br i1 %c, label %run, label %end

run:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double, [-1,40]:Float@double, [-1,48]:Float@double, [-1,56]:Float@double, [-1,64]:Float@double, [-1,72]:Float@double}" %0, ptr align 8 "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double, [-1,32]:Float@double, [-1,40]:Float@double, [-1,48]:Float@double, [-1,56]:Float@double, [-1,64]:Float@double, [-1,72]:Float@double}" %1, i64 "enzyme_type"="{[-1]:Integer}" 80, i1 false)
  br label %end

end:
  ret void
}

define void @dsquare(ptr %x, ptr %dx, ptr %y, ptr %dy) {
entry:
  tail call void (void (i1, ptr, ptr)*, ...) @__enzyme_autodiff(void (i1, ptr, ptr)* nonnull @cpy, i1 1, ptr %x, ptr %dx, ptr %y, ptr %dy)
  ret void
}
