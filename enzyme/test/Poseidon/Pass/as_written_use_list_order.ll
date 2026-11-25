; A profile-use compile whose every site is left alone emits the module as a
; build without the plugin does, the order of each value's uses included: a
; block's predecessors are listed, and visited by the passes after, in the
; order the module's reader gave them.
; RUN: rm -rf %t && mkdir %t
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %s -S -emit-llvm -o %t/plain.ll
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %s %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/c -S -emit-llvm -o %t/use.ll
; RUN: grep -v -e llvm.metadata -e '^$' %t/plain.ll > %t/plain.cmp
; RUN: grep -v -e llvm.metadata -e '^$' %t/use.ll > %t/use.cmp
; RUN: diff %t/plain.cmp %t/use.cmp
; RUN: FileCheck %s < %t/use.ll

; CHECK: join: {{.*}}; preds = %pos, %neg

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [6 x i8] c"t.cpp\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @site, ptr @.str, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

define double @site(double %x) #0 {
  %m = fmul double %x, %x
  %a = fadd double %m, 1.0
  ret double %a
}

define i32 @pick(i32 %n) #0 {
entry:
  %c = icmp sgt i32 %n, 0
  br i1 %c, label %pos, label %neg
pos:
  br label %join
neg:
  br label %join
join:
  %r = phi i32 [ 1, %pos ], [ 2, %neg ]
  ret i32 %r
  uselistorder label %join, { 1, 0 }
}

attributes #0 = { noinline nounwind "target-cpu"="x86-64" }
