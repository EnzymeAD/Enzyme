; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.SubArray = type { i8*, [1 x [2 x i64]], i64, i64 }

define "enzyme_type"="{[0]:Pointer, [8]:Integer, [9]:Integer, [10]:Integer, [11]:Integer, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer, [16]:Integer, [17]:Integer, [18]:Integer, [19]:Integer, [20]:Integer, [21]:Integer, [22]:Integer, [23]:Integer, [24]:Integer, [25]:Integer, [26]:Integer, [27]:Integer, [28]:Integer, [29]:Integer, [30]:Integer, [31]:Integer, [32]:Integer, [33]:Integer, [34]:Integer, [35]:Integer, [36]:Integer, [37]:Integer, [38]:Integer, [39]:Integer}" %struct.SubArray @foo(i8* %p) {
entry:
  %val = insertvalue %struct.SubArray { i8* undef, [1 x [2 x i64]] [[2 x i64] [i64 1, i64 2]], i64 0, i64 1 }, i8* %p, 0
  ret %struct.SubArray %val
}

declare { i8*, %struct.SubArray, %struct.SubArray } @__enzyme_augmentfwd(...)

define void @test(i8* %p, i8* %dp) {
entry:
  %res = call { i8*, %struct.SubArray, %struct.SubArray } (...) @__enzyme_augmentfwd(metadata !"enzyme_dup", %struct.SubArray (i8*)* @foo, metadata !"enzyme_dup", i8* %p, i8* %dp)
  ret void
}

; CHECK: define internal { {{(i8\*|ptr)}}, %struct.SubArray, %struct.SubArray } @augmented_foo(i8* %p, i8* %"p'")
; CHECK: entry:
; CHECK:   %"val'ipiv" = insertvalue %struct.SubArray { {{(i8\*|ptr)}} null, [1 x [2 x i64]] {{.*}}i64 1, i64 2{{.*}}, i64 0, i64 1 }, {{(i8\*|ptr)}} %"p'", 0
