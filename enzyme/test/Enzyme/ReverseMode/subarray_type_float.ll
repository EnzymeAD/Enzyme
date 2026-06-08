; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.SubArray = type { ptr, [1 x [2 x i64]], i64, i64 }

define "enzyme_type"="{[0]:Pointer, [8]:Float@double, [16]:Float@double, [24]:Integer, [32]:Integer}" %struct.SubArray @foo(ptr %p) {
entry:
  %val = insertvalue %struct.SubArray { ptr null, [1 x [2 x i64]] [[2 x i64] [i64 1, i64 2]], i64 0, i64 1 }, ptr %p, 0
  ret %struct.SubArray %val
}

declare { ptr, %struct.SubArray, %struct.SubArray } @__enzyme_augmentfwd(...)

define void @test(ptr %p, ptr %dp, ptr %out_shadow) {
entry:
  %res = call { ptr, %struct.SubArray, %struct.SubArray } (...) @__enzyme_augmentfwd(ptr @foo, metadata !"enzyme_dup", ptr %p, ptr %dp)
  %shadow_ret = extractvalue { ptr, %struct.SubArray, %struct.SubArray } %res, 2
  store %struct.SubArray %shadow_ret, ptr %out_shadow, align 8
  ret void
}

; CHECK: define internal { ptr, %struct.SubArray, %struct.SubArray } @augmented_foo(ptr %p, ptr %"p'")
; CHECK: entry:
; CHECK:   %"val'ipiv" = insertvalue %struct.SubArray { ptr null, [1 x [2 x i64]] zeroinitializer, i64 0, i64 1 }, ptr %"p'", 0
