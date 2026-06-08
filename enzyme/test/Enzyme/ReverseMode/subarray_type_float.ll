; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.SubArray = type { ptr, [1 x [2 x i64]], i64, i64 }

define void @foo(ptr %p, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Integer, [-1,32]:Integer}" %out) {
entry:
  %val = insertvalue %struct.SubArray { ptr null, [1 x [2 x i64]] [[2 x i64] [i64 1, i64 2]], i64 0, i64 1 }, ptr %p, 0
  store %struct.SubArray %val, ptr %out, align 8
  ret void
}

declare { ptr } @__enzyme_augmentfwd(...)

define void @test(ptr %p, ptr %dp, ptr %out, ptr %dout) {
entry:
  %res = call { ptr } (...) @__enzyme_augmentfwd(ptr @foo, metadata !"enzyme_dup", ptr %p, ptr %dp, metadata !"enzyme_dup", ptr %out, ptr %dout)
  ret void
}

; CHECK: define internal ptr @augmented_foo(ptr %p, ptr %"p'", ptr "enzyme_type"={{.*}} %out, ptr "enzyme_type"={{.*}} %"out'")
; CHECK: entry:
; CHECK:   %"val'ipiv" = insertvalue %struct.SubArray { ptr null, [1 x [2 x i64]] zeroinitializer, i64 0, i64 1 }, ptr %"p'", 0
