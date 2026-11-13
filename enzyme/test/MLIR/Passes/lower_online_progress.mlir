// RUN: %eopt %s --lower-enzyme-online-progress --canonicalize | FileCheck %s

// Verify that enzyme.online_progress is lowered to scf/arith/memref ops.
// The test uses snaps=2 (the smallest interesting case): only the r=1 takeshot,
// r=2 advance case 2 (capo = ch[1] + incr), and r=2 takeshot case 2
// (ch[1] = capo; incr++) branches are reachable after constant-folding on snaps.

module {
  func.func @test_one_step(
      %check   : index, %capo    : index, %oldcapo : index,
      %t       : index, %incr    : index, %iter    : index,
      %oldind  : index, %ind     : index, %old_f   : index,
      %offset  : index,
      %ch      : memref<2xindex>,
      %ord_ch  : memref<2xindex>,
      %num_rep : memref<2xindex>
  ) -> (index, index, index, index, index, index, index, index, index, index, index) {
    %snaps = arith.constant 2 : index
    %action, %nc, %nca, %noc, %nt, %ni, %nit, %noi, %nind, %nof, %noff =
        enzyme.online_progress
          %check, %capo, %oldcapo, %t, %incr, %iter, %oldind, %ind, %old_f, %offset,
          %snaps, %ch, %ord_ch, %num_rep
          : memref<2xindex>, memref<2xindex>, memref<2xindex>
    return %action, %nc, %nca, %noc, %nt, %ni, %nit, %noi, %nind, %nof, %noff
        : index, index, index, index, index, index, index, index, index, index, index
  }
}

// The enzyme.online_progress op must be gone after lowering.
// CHECK-NOT: enzyme.online_progress

// CHECK-LABEL: func.func @test_one_step(

// The top-level branch checks whether check < 0 (r=1 takeshot) or not.
// CHECK:         arith.cmpi slt
// CHECK:         scf.if

// r=1 takeshot branch (snaps=2 < 4): fills num_rep[*] = 2 via a for loop.
// CHECK:         scf.for
// CHECK:         memref.store

// r=2 advance case 2: capo_new = ch[1] + incr → load ch[1].
// CHECK:         memref.load

// r=2 takeshot case 2: ch[1] = capo (store to memref).
// CHECK:         memref.store

// The function returns 11 index values (action + 10 updated SM scalars).
// CHECK:         return
// CHECK-SAME:    index, index, index, index, index, index, index, index, index, index, index
