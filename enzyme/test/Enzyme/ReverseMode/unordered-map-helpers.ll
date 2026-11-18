; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

@.str = private unnamed_addr constant [8 x i8] c"alpha_x\00", align 1

declare i32 @bcmp(ptr, ptr, i64)
declare i64 @_ZSt11_Hash_bytesPKvmm(ptr, i64, i64)
declare i64 @_ZSt15_Fnv_hash_bytesPKvmm(ptr, i64, i64)
declare i64 @_ZNKSt8__detail20_Prime_rehash_policy11_M_next_bktEm(ptr, i64)
declare ptr @malloc(i64)
declare void @free(ptr)

define double @f(double %x, ptr %key, ptr %policy) {
entry:
  %h1 = call i64 @_ZSt11_Hash_bytesPKvmm(ptr %key, i64 7, i64 0)
  %h2 = call i64 @_ZSt15_Fnv_hash_bytesPKvmm(ptr %key, i64 7, i64 %h1)
  %bkt = call i64 @_ZNKSt8__detail20_Prime_rehash_policy11_M_next_bktEm(ptr %policy, i64 %h2)
  %cmp = call i32 @bcmp(ptr %key, ptr @.str, i64 7)
  %buf = call ptr @malloc(i64 8)
  store double %x, ptr %buf, align 8
  %v = load double, ptr %buf, align 8
  %eq = icmp eq i32 %cmp, 0
  %nz = icmp ne i64 %bkt, 0
  %both = and i1 %eq, %nz
  %sq = fmul double %v, %v
  %res = select i1 %both, double %sq, double %v
  call void @free(ptr %buf)
  ret double %res
}

declare double @__enzyme_autodiff(...)

define double @df(double %x, ptr %key, ptr %policy) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x, metadata !"enzyme_const", ptr %key, metadata !"enzyme_const", ptr %policy)
  ret double %r
}

; CHECK: define internal { double } @diffef(double %x, ptr {{.*}}%key, ptr %policy, double %differeturn)
; CHECK: call i64 @_ZSt11_Hash_bytesPKvmm(ptr %key, i64 7, i64 0)
; CHECK: call i64 @_ZSt15_Fnv_hash_bytesPKvmm(ptr %key, i64 7, i64 %h1)
; CHECK: call i64 @_ZNKSt8__detail20_Prime_rehash_policy11_M_next_bktEm(ptr %policy, i64 %h2)
; CHECK: call i32 @bcmp(ptr %key, ptr @.str, i64 7)
; CHECK: ret { double }
