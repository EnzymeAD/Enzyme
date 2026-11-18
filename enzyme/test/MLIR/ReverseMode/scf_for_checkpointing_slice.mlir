// RUN: %eopt %s --enzyme-wrap="infn=periodic_unused_inactive outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=PERIODIC_DEAD
// RUN: %eopt %s --enzyme-wrap="infn=binomial_unused_inactive outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=BINOMIAL_DEAD
// RUN: %eopt %s --enzyme-wrap="infn=affine_unused_inactive outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=AFFINE_DEAD
// RUN: %eopt %s --enzyme-wrap="infn=periodic_coupled_inactive outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=COUPLED
// RUN: %eopt %s --enzyme-wrap="infn=periodic_linear_acc outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=PERIODIC_ZERO
// RUN: %eopt %s --enzyme-wrap="infn=binomial_linear_acc outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=BINOMIAL_ZERO
// RUN: %eopt %s --enzyme-wrap="infn=affine_linear_acc outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=AFFINE_ZERO
// RUN: %eopt %s --enzyme-wrap="infn=periodic_local_unwritten outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=LOCAL_RO
// RUN: %eopt %s --enzyme-wrap="infn=periodic_written_after outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=WRITTEN_AFTER
// RUN: %eopt %s --enzyme-wrap="infn=periodic_freed_after outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=FREED_AFTER
// RUN: %eopt %s --enzyme-wrap="infn=periodic_written_outer outfn= argTys=enzyme_dup retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=WRITTEN_OUTER
// RUN: %eopt %s --enzyme-wrap="infn=periodic_aliasing_write outfn= argTys=enzyme_dup,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=ALIAS_WRITE
// RUN: %eopt %s --enzyme-wrap="infn=periodic_unused_memref outfn= argTys=enzyme_dup,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math | FileCheck %s --check-prefix=UNUSED_MEMREF

module {

  // 1. Unused inactive iter_arg (f64) is omitted from periodic checkpoints.
  func.func @periodic_unused_inactive(%x: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %dead0 = arith.constant 1.0 : f64
    %inc = arith.constant 2.0 : f64
    %res:2 = scf.for %i = %c0 to %c9 step %c1 iter_args(%a = %x, %d = %dead0) -> (f32, f64) {
      %mul = arith.mulf %a, %a : f32
      %d2 = arith.addf %d, %inc : f64
      scf.yield %mul, %d2 : f32, f64
    } {enzyme.enable_checkpointing = true}
    return %res#0 : f32
  }
  // PERIODIC_DEAD-LABEL: func.func @periodic_unused_inactive(
  // PERIODIC_DEAD-NOT: memref.alloc() : memref<3xf64>
  // PERIODIC_DEAD:     %[[CKPT:.+]] = memref.alloc() : memref<3xf32>
  // PERIODIC_DEAD-NOT: memref.alloc() : memref<3xf64>
  // PERIODIC_DEAD:     scf.for {{.*}} iter_args({{.*}}) -> (f32)
  // PERIODIC_DEAD-NOT: arith.addf {{.*}} : f64
  // PERIODIC_DEAD:     memref.dealloc %[[CKPT]] : memref<3xf32>

  // 2. Unused inactive iter_arg (f64) is omitted from binomial checkpoints.
  func.func @binomial_unused_inactive(%x: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %dead0 = arith.constant 1.0 : f64
    %inc = arith.constant 2.0 : f64
    %res:2 = scf.for %i = %c0 to %c9 step %c1 iter_args(%a = %x, %d = %dead0) -> (f32, f64) {
      %mul = arith.mulf %a, %a : f32
      %d2 = arith.addf %d, %inc : f64
      scf.yield %mul, %d2 : f32, f64
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 3 : i64}
    return %res#0 : f32
  }
  // BINOMIAL_DEAD-LABEL: func.func @binomial_unused_inactive(
  // BINOMIAL_DEAD-NOT: memref.alloc() : memref<3xf64>
  // BINOMIAL_DEAD:     %[[STATE:.+]] = memref.alloc() : memref<3xf32>
  // BINOMIAL_DEAD:     %[[IDX:.+]] = memref.alloc() : memref<3xindex>
  // BINOMIAL_DEAD-NOT: memref.alloc() : memref<3xf64>
  // BINOMIAL_DEAD:     scf.while ({{.*}}) : (index, index, f32) -> (index, index, f32)
  // BINOMIAL_DEAD:     memref.dealloc %[[STATE]] : memref<3xf32>
  // BINOMIAL_DEAD:     memref.dealloc %[[IDX]] : memref<3xindex>

  // 3. Unused inactive iter_arg (f64) is omitted from affine.for checkpoints.
  func.func @affine_unused_inactive(%x: f32) -> f32 {
    %dead0 = arith.constant 1.0 : f64
    %inc = arith.constant 2.0 : f64
    %res:2 = affine.for %i = 0 to 9 iter_args(%a = %x, %d = %dead0) -> (f32, f64) {
      %mul = arith.mulf %a, %a : f32
      %d2 = arith.addf %d, %inc : f64
      affine.yield %mul, %d2 : f32, f64
    } {enzyme.enable_checkpointing = true}
    return %res#0 : f32
  }
  // AFFINE_DEAD-LABEL: func.func @affine_unused_inactive(
  // AFFINE_DEAD-NOT: memref.alloc() : memref<3xf64>
  // AFFINE_DEAD:     %[[CKPT:.+]] = memref.alloc() : memref<3xf32>
  // AFFINE_DEAD-NOT: memref.alloc() : memref<3xf64>
  // AFFINE_DEAD:     affine.for {{.*}} iter_args({{.*}}) -> (f32)
  // AFFINE_DEAD:     memref.dealloc %[[CKPT]] : memref<3xf32>

  // 4. Counterexample: inactive iter_arg %s feeding an active mulf %a, %s MUST
  //    be checkpointed (here %a is linear in %s so only %s is needed!).
  func.func @periodic_coupled_inactive(%x: f64) -> f64 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %s0 = arith.constant 1.0 : f64
    %inc = arith.constant 0.5 : f64
    %res:2 = scf.for %i = %c0 to %c9 step %c1 iter_args(%a = %x, %s = %s0) -> (f64, f64) {
      %mul = arith.mulf %a, %s : f64
      %s2 = arith.addf %s, %inc : f64
      scf.yield %mul, %s2 : f64, f64
    } {enzyme.enable_checkpointing = true}
    return %res#0 : f64
  }
  // COUPLED-LABEL: func.func @periodic_coupled_inactive(
  // COUPLED:     %[[CKPT:.+]] = memref.alloc() : memref<3xf64>
  // COUPLED-NOT: memref.alloc() : memref<3xf64>
  // COUPLED:     scf.for {{.*}} iter_args({{.*}}) -> (f64, f64)
  // Replay only carries %s (the inactive factor needed for da += dmul * s).
  // COUPLED:     scf.for {{.*}} iter_args({{.*}}) -> (f64) {
  // COUPLED:       arith.addf {{.*}} : f64
  // COUPLED:     memref.dealloc %[[CKPT]] : memref<3xf64>

  // 5. Linear active accumulator (arith.addf) needs ZERO checkpointed iter_args
  //    and read-only %m needs ZERO clones/copies (periodic).
  func.func @periodic_linear_acc(%m: memref<9xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %z = arith.constant 0.0 : f32
    %sum = scf.for %i = %c0 to %c9 step %c1 iter_args(%acc = %z) -> (f32) {
      %v = memref.load %m[%i] : memref<9xf32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    } {enzyme.enable_checkpointing = true}
    return %sum : f32
  }
  // PERIODIC_ZERO-LABEL: func.func @periodic_linear_acc(
  // PERIODIC_ZERO-SAME:    %[[ARG0:.+]]: memref<9xf32>, %[[ARG1:.+]]: memref<9xf32>, %[[ARG2:.+]]: f32)
  // PERIODIC_ZERO-NOT: memref.alloc
  // PERIODIC_ZERO-NOT: memref.copy
  // PERIODIC_ZERO:     scf.for %{{.+}} = %[[C0:.+]] to %[[C3:.+]] step %[[C1:.+]] iter_args(%[[OUT_ADJ:.+]] = %[[ARG2]]) -> (f32)
  // PERIODIC_ZERO:       scf.for %{{.+}} = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%{{.+}} = %[[OUT_ADJ]]) -> (f32)
  // PERIODIC_ZERO:         memref.load %[[ARG1]][
  // PERIODIC_ZERO:         memref.store {{.*}}, %[[ARG1]][

  // 6. Linear active accumulator needs ZERO state checkpoint buffer in binomial mode.
  func.func @binomial_linear_acc(%m: memref<9xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %z = arith.constant 0.0 : f32
    %sum = scf.for %i = %c0 to %c9 step %c1 iter_args(%acc = %z) -> (f32) {
      %v = memref.load %m[%i] : memref<9xf32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    } {enzyme.enable_checkpointing = true,
       enzyme.binomial_checkpointing,
       enzyme.checkpoint_period = 3 : i64}
    return %sum : f32
  }
  // BINOMIAL_ZERO-LABEL: func.func @binomial_linear_acc(
  // BINOMIAL_ZERO-SAME:    %[[ARG0:.+]]: memref<9xf32>, %[[ARG1:.+]]: memref<9xf32>, %[[ARG2:.+]]: f32)
  // BINOMIAL_ZERO-NOT: memref.alloc() : memref<3xf32>
  // BINOMIAL_ZERO-NOT: memref.copy
  // BINOMIAL_ZERO:     %[[IDX:.+]] = memref.alloc() : memref<3xindex>
  // BINOMIAL_ZERO:     scf.while ({{.*}}) : (index, index) -> (index, index)
  // BINOMIAL_ZERO:       memref.load %[[ARG1]][
  // BINOMIAL_ZERO:       memref.store {{.*}}, %[[ARG1]][
  // BINOMIAL_ZERO:     memref.dealloc %[[IDX]] : memref<3xindex>

  // 7. Linear active accumulator in affine.for needs ZERO iter_arg checkpoints.
  func.func @affine_linear_acc(%m: memref<9xf32>) -> f32 {
    %z = arith.constant 0.0 : f32
    %sum = affine.for %i = 0 to 9 iter_args(%acc = %z) -> (f32) {
      %v = memref.load %m[%i] : memref<9xf32>
      %next = arith.addf %acc, %v : f32
      affine.yield %next : f32
    } {enzyme.enable_checkpointing = true}
    return %sum : f32
  }
  // AFFINE_ZERO-LABEL: func.func @affine_linear_acc(
  // AFFINE_ZERO-SAME:    %[[ARG0:.+]]: memref<9xf32>, %[[ARG1:.+]]: memref<9xf32>, %[[ARG2:.+]]: f32)
  // AFFINE_ZERO-NOT: memref.alloc
  // AFFINE_ZERO-NOT: memref.copy
  // AFFINE_ZERO:     affine.for %{{.+}} = 0 to 3 iter_args(%[[OUT_ADJ:.+]] = %[[ARG2]]) -> (f32)
  // AFFINE_ZERO:       affine.for %{{.+}} = 0 to 3 iter_args(%{{.+}} = %[[OUT_ADJ]]) -> (f32)
  // AFFINE_ZERO:         memref.load %[[ARG1]][
  // AFFINE_ZERO:         memref.store {{.*}}, %[[ARG1]][

  // 8. Local memref.alloc initialized before the loop and never written after is not cloned.
  func.func @periodic_local_unwritten(%x: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %buf = memref.alloc() : memref<f32>
    memref.store %x, %buf[] : memref<f32>
    %one = arith.constant 1.0 : f32
    %res = scf.for %i = %c0 to %c9 step %c1 iter_args(%a = %one) -> (f32) {
      %v = memref.load %buf[] : memref<f32>
      %m = arith.mulf %a, %v : f32
      scf.yield %m : f32
    } {enzyme.enable_checkpointing = true}
    return %res : f32
  }
  // LOCAL_RO-LABEL: func.func @periodic_local_unwritten(
  // LOCAL_RO-NOT: memref.copy

  // 9. Counterexample: memref written AFTER the loop MUST still be cloned.
  func.func @periodic_written_after(%m: memref<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %res = scf.for %i = %c0 to %c9 step %c1 iter_args(%a = %one) -> (f32) {
      %v = memref.load %m[] : memref<f32>
      %mul = arith.mulf %a, %v : f32
      scf.yield %mul : f32
    } {enzyme.enable_checkpointing = true}
    memref.store %zero, %m[] : memref<f32>
    return %res : f32
  }
  // WRITTEN_AFTER-LABEL: func.func @periodic_written_after(
  // WRITTEN_AFTER-SAME:    %[[ARG0:[^:]+]]: memref<f32>,
  // WRITTEN_AFTER: memref.copy %[[ARG0]],

  // 10. Counterexample: memref deallocated AFTER the loop MUST still be cloned.
  func.func @periodic_freed_after(%x: f32, %buf: memref<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %res = scf.for %i = %c0 to %c9 step %c1 iter_args(%a = %x) -> (f32) {
      %v = memref.load %buf[] : memref<f32>
      %mul = arith.mulf %a, %v : f32
      scf.yield %mul : f32
    } {enzyme.enable_checkpointing = true}
    memref.dealloc %buf : memref<f32>
    return %res : f32
  }
  // FREED_AFTER-LABEL: func.func @periodic_freed_after(
  // FREED_AFTER-SAME:    %{{[^:]+}}: f32, %[[BUF:[^:]+]]: memref<f32>,
  // FREED_AFTER: memref.copy %[[BUF]],

  // 11. Counterexample: memref written in enclosing outer loop before inner loop MUST be cloned.
  func.func @periodic_written_outer(%m: memref<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c9 = arith.constant 9 : index
    %one = arith.constant 1.0 : f32
    %outer = scf.for %j = %c0 to %c2 step %c1 iter_args(%oa = %one) -> (f32) {
      memref.store %oa, %m[] : memref<f32>
      %inner = scf.for %i = %c0 to %c9 step %c1 iter_args(%ia = %oa) -> (f32) {
        %v = memref.load %m[] : memref<f32>
        %mul = arith.mulf %ia, %v : f32
        scf.yield %mul : f32
      } {enzyme.enable_checkpointing = true}
      scf.yield %inner : f32
    }
    return %outer : f32
  }
  // WRITTEN_OUTER-LABEL: func.func @periodic_written_outer(
  // WRITTEN_OUTER-SAME:    %[[ARG0:[^:]+]]: memref<f32>,
  // WRITTEN_OUTER: memref.copy %[[ARG0]],

  // 12. Counterexample: unannotated aliasing arg written inside loop MUST force clone of read arg.
  func.func @periodic_aliasing_write(%a: memref<f32>, %b: memref<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %one = arith.constant 1.0 : f32
    %z = arith.constant 2.0 : f32
    %res = scf.for %i = %c0 to %c9 step %c1 iter_args(%it = %one) -> (f32) {
      %v = memref.load %a[] : memref<f32>
      memref.store %z, %b[] : memref<f32>
      %mul = arith.mulf %it, %v : f32
      scf.yield %mul : f32
    } {enzyme.enable_checkpointing = true}
    return %res : f32
  }
  // ALIAS_WRITE-LABEL: func.func @periodic_aliasing_write(
  // ALIAS_WRITE-SAME:    %[[ARG0:[^:]+]]: memref<f32>,
  // ALIAS_WRITE: memref.copy %[[ARG0]],

  // 13. Outside memref used ONLY by an inactive dead op is not cloned even if written elsewhere.
  func.func @periodic_unused_memref(%active: memref<f32>, %unused: memref<f32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %one = arith.constant 1.0 : f32
    %d0 = arith.constant 0.0 : f32
    %res:2 = scf.for %i = %c0 to %c9 step %c1 iter_args(%it = %one, %dead = %d0) -> (f32, f32) {
      %v = memref.load %active[] : memref<f32>
      %mul = arith.mulf %it, %v : f32
      memref.store %mul, %active[] : memref<f32>
      %u = memref.load %unused[] : memref<f32>
      %d1 = arith.addf %dead, %u : f32
      scf.yield %mul, %d1 : f32, f32
    } {enzyme.enable_checkpointing = true}
    memref.store %one, %unused[] : memref<f32>
    return %res#0 : f32
  }
  // UNUSED_MEMREF-LABEL: func.func @periodic_unused_memref(
  // UNUSED_MEMREF-SAME:    %[[ACTIVE:[^:]+]]: memref<f32>, %{{[^:]+}}: memref<f32>, %[[UNUSED:[^:]+]]: memref<f32>,
  // UNUSED_MEMREF-NOT: memref.copy %[[UNUSED]]
  // UNUSED_MEMREF:     memref.copy %[[ACTIVE]],
  // UNUSED_MEMREF-NOT: memref.copy %[[UNUSED]]

}
