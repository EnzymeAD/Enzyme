; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.SubArray = type { i8*, [1 x [2 x i64]], i64, i64 }

define void @foo(i8* %p, %struct.SubArray* "enzyme_type"="{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Integer, [-1,17]:Integer, [-1,18]:Integer, [-1,19]:Integer, [-1,20]:Integer, [-1,21]:Integer, [-1,22]:Integer, [-1,23]:Integer, [-1,24]:Integer, [-1,25]:Integer, [-1,26]:Integer, [-1,27]:Integer, [-1,28]:Integer, [-1,29]:Integer, [-1,30]:Integer, [-1,31]:Integer, [-1,32]:Integer, [-1,33]:Integer, [-1,34]:Integer, [-1,35]:Integer, [-1,36]:Integer, [-1,37]:Integer, [-1,38]:Integer, [-1,39]:Integer}" %out) {
entry:
  %val = insertvalue %struct.SubArray { i8* null, [1 x [2 x i64]] [[2 x i64] [i64 1, i64 2]], i64 0, i64 1 }, i8* %p, 0
  store %struct.SubArray %val, %struct.SubArray* %out, align 8
  ret void
}

declare { i8* } @__enzyme_augmentfwd(...)

define void @test(i8* %p, i8* %dp, %struct.SubArray* %out, %struct.SubArray* %dout) {
entry:
  %res = call { i8* } (...) @__enzyme_augmentfwd(void (i8*, %struct.SubArray*)* @foo, metadata !"enzyme_dup", i8* %p, i8* %dp, metadata !"enzyme_dup", %struct.SubArray* %out, %struct.SubArray* %dout)
  ret void
}

; CHECK: define internal {{(i8\*|ptr)}} @augmented_foo(i8* %p, i8* %"p'", %struct.SubArray* "enzyme_type"={{.*}} %out, %struct.SubArray* "enzyme_type"={{.*}} %"out'")
; CHECK: entry:
; CHECK:   %"val'ipiv" = insertvalue %struct.SubArray { {{(i8\*|ptr)}} null, [1 x [2 x i64]] {{.*}}i64 1, i64 2{{.*}}, i64 0, i64 1 }, {{(i8\*|ptr)}} %"p'", 0
