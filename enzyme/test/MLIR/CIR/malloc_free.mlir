// RUN: %eopt --enzyme --canonicalize %s | FileCheck %s

// The C and C++ allocators have no body to differentiate. An active
// allocation gets a shadow of the same size: malloc's and operator new's are
// zeroed, calloc's is zero already. In forward mode the shadow is freed with
// the primal, and resized with it by realloc. In reverse mode the reverse
// sweep still uses the shadow after the primal free, so it is freed when the
// sweep gets back to the allocation, and only if every path from the
// allocation frees it (a path where the primal is null needs no free);
// otherwise it is left allocated. realloc's shadow is a fresh allocation the
// adjoints of the copy are added back from, and realloc frees the old one.

!u64i = !cir.int<u, 64>
!void = !cir.void
module {
  cir.func private @malloc(!u64i) -> !cir.ptr<!void>
  cir.func private @calloc(!u64i, !u64i) -> !cir.ptr<!void>
  cir.func private @free(!cir.ptr<!void>)
  cir.func private @realloc(!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
  cir.func private @_Znwm(!u64i) -> !cir.ptr<!void>
  cir.func private @_ZdlPvm(!cir.ptr<!void>, !u64i)

  // y = x * x, through a heap buffer
  cir.func @square(%x: !cir.double) -> !cir.double {
    %n = cir.const #cir.int<8> : !u64i
    %m = cir.call @malloc(%n) : (!u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %m : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.call @free(%m) : (!cir.ptr<!void>) -> ()
    cir.return %y : !cir.double
  }

  // The same, from calloc, and never freed
  cir.func @square_kept(%x: !cir.double) -> !cir.double {
    %one = cir.const #cir.int<1> : !u64i
    %n = cir.const #cir.int<8> : !u64i
    %m = cir.call @calloc(%one, %n) : (!u64i, !u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %m : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.return %y : !cir.double
  }

  // Freed on one path only
  cir.func @square_maybe(%x: !cir.double, %c: !cir.bool) -> !cir.double {
    %n = cir.const #cir.int<8> : !u64i
    %m = cir.call @malloc(%n) : (!u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %m : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.brcond %c ^free, ^done
  ^free:
    cir.call @free(%m) : (!cir.ptr<!void>) -> ()
    cir.br ^done
  ^done:
    cir.return %y : !cir.double
  }

  // What clang makes of `new double` and `delete p`: operator delete is only
  // called when p is not null.
  cir.func @object(%x: !cir.double) -> !cir.double {
    %null = cir.const #cir.ptr<null> : !cir.ptr<!cir.double>
    %n = cir.const #cir.int<8> : !u64i
    %m = cir.call @_Znwm(%n) : (!u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %m : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    %nonnull = cir.cmp ne %t, %null : !cir.ptr<!cir.double>
    cir.brcond %nonnull ^delete, ^done
  ^delete:
    %v = cir.cast bitcast %t : !cir.ptr<!cir.double> -> !cir.ptr<!void>
    cir.call @_ZdlPvm(%v, %n) : (!cir.ptr<!void>, !u64i) -> ()
    cir.br ^done
  ^done:
    cir.return %y : !cir.double
  }

  // y = x * x, through a buffer grown by realloc
  cir.func @grown(%x: !cir.double) -> !cir.double {
    %n = cir.const #cir.int<8> : !u64i
    %n2 = cir.const #cir.int<16> : !u64i
    %m = cir.call @malloc(%n) : (!u64i) -> !cir.ptr<!void>
    %r = cir.call @realloc(%m, %n2) : (!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
    %t = cir.cast bitcast %r : !cir.ptr<!void> -> !cir.ptr<!cir.double>
    %sq = cir.fmul %x, %x : !cir.double
    cir.store %sq, %t : !cir.double, !cir.ptr<!cir.double>
    %y = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.call @free(%r) : (!cir.ptr<!void>) -> ()
    cir.return %y : !cir.double
  }

  cir.func @fwd(%x: !cir.double, %dx: !cir.double) -> !cir.double {
    %r = enzyme.fwddiff @square(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev(%x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @square(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev_kept(%x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @square_kept(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev_maybe(%x: !cir.double, %c: !cir.bool, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @square_maybe(%x, %c, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.bool, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev_object(%x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @object(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @fwd_grown(%x: !cir.double, %dx: !cir.double) -> !cir.double {
    %r = enzyme.fwddiff @grown(%x, %dx) <{ activity=[#enzyme.activity<enzyme_dup>], ret_activity=[#enzyme.activity<enzyme_dupnoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev_grown(%x: !cir.double, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @grown(%x, %seed) <{ activity=[#enzyme.activity<enzyme_active>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// Forward mode: a zeroed shadow next to the primal, freed with it.
// CHECK-LABEL: cir.func private @fwddiffesquare(
// CHECK:         %[[DM:.+]] = cir.call @malloc(%[[N:.+]]) : (!u64i) -> !cir.ptr<!void>
// CHECK-NEXT:    cir.libc.memset %[[N]] bytes at %[[DM]] to
// CHECK-NEXT:    %[[M:.+]] = cir.call @malloc(%[[N]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:         cir.call @free(%[[DM]])
// CHECK-NEXT:    cir.call @free(%[[M]])
// CHECK-NEXT:    cir.return

// Reverse mode: the primal is freed in the forward sweep, the shadow once the
// reverse sweep is back at the malloc.
// CHECK-LABEL: cir.func private @diffesquare(
// CHECK:         %[[CACHE:.+]] = "enzyme.init"() : () -> !enzyme.Cache<!cir.ptr<!void>>
// CHECK:         %[[DM:.+]] = cir.call @malloc(%[[N:.+]]) : (!u64i) -> !cir.ptr<!void>
// CHECK-NEXT:    cir.libc.memset %[[N]] bytes at %[[DM]] to
// CHECK-NEXT:    "enzyme.push"(%[[CACHE]], %[[DM]])
// CHECK-NEXT:    %[[M:.+]] = cir.call @malloc(%[[N]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:         cir.call @free(%[[M]])
// CHECK-NOT:     cir.call @free
// CHECK:         %[[DMP:.+]] = "enzyme.pop"(%[[CACHE]])
// CHECK-NEXT:    cir.call @free(%[[DMP]])
// CHECK-NEXT:    "enzyme.get"
// CHECK-NEXT:    cir.return

// Never freed: neither is the shadow. calloc's needs no memset.
// CHECK-LABEL: cir.func private @diffesquare_kept(
// CHECK:         cir.call @calloc(
// CHECK-NEXT:    cir.call @calloc(
// CHECK-NOT:     cir.libc.memset
// CHECK-NOT:     cir.call @free
// CHECK:         cir.return

// Freed on one path only: the primal free stays where it was, and the shadow
// is not freed.
// CHECK-LABEL: cir.func private @diffesquare_maybe(
// CHECK:         cir.call @malloc(
// CHECK-NEXT:    cir.libc.memset
// CHECK-NEXT:    cir.call @malloc(
// CHECK:         cir.call @free(
// CHECK-NOT:     cir.call @free
// CHECK:         cir.return

// operator new is zeroed like malloc. The null test in front of the delete
// still counts as freeing on every path, and the shadow is freed with the
// same sized operator delete.
// CHECK-LABEL: cir.func private @diffeobject(
// CHECK:         %[[CACHE:.+]] = "enzyme.init"() : () -> !enzyme.Cache<!cir.ptr<!void>>
// CHECK:         %[[DM:.+]] = cir.call @_Znwm(%[[N:.+]]) : (!u64i) -> !cir.ptr<!void>
// CHECK-NEXT:    cir.libc.memset %[[N]] bytes at %[[DM]] to
// CHECK-NEXT:    "enzyme.push"(%[[CACHE]], %[[DM]])
// CHECK-NEXT:    %[[M:.+]] = cir.call @_Znwm(%[[N]]) : (!u64i) -> !cir.ptr<!void>
// CHECK:         cir.brcond
// CHECK:         cir.call @_ZdlPvm(%{{.+}}, %[[N]])
// CHECK:         %[[DMP:.+]] = "enzyme.pop"(%[[CACHE]])
// CHECK-NEXT:    cir.call @_ZdlPvm(%[[DMP]], %[[N]])
// CHECK-NEXT:    "enzyme.get"
// CHECK-NEXT:    cir.return

// Forward mode: realloc resizes the shadow along with the primal.
// CHECK-LABEL: cir.func private @fwddiffegrown(
// CHECK:         %[[DM:.+]] = cir.call @malloc(
// CHECK-NEXT:    cir.libc.memset
// CHECK-NEXT:    %[[M:.+]] = cir.call @malloc(
// CHECK-NEXT:    %[[DR:.+]] = cir.call @realloc(%[[DM]], %[[N2:.+]]) : (!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
// CHECK-NEXT:    %[[R:.+]] = cir.call @realloc(%[[M]], %[[N2]]) : (!cir.ptr<!void>, !u64i) -> !cir.ptr<!void>
// CHECK:         cir.call @free(%[[DR]])
// CHECK-NEXT:    cir.call @free(%[[R]])
// CHECK-NEXT:    cir.return

// Reverse mode: realloc's shadow is realloc(NULL, n), zeroed. The old size
// comes from the malloc. The reverse sweep adds the adjoints of the copied
// doubles back into the old shadow, frees the new one, and frees the old one
// at the malloc, as the realloc frees what it returned.
// CHECK-LABEL: cir.func private @diffegrown(
// CHECK:         %[[SIZE:.+]] = cir.alloca "realloc.size"
// CHECK:         %[[DM:.+]] = cir.call @malloc(%[[N:[^)]+]])
// CHECK-NEXT:    cir.libc.memset %[[N]] bytes at %[[DM]]
// CHECK-NEXT:    "enzyme.push"(%[[DMCACHE:.+]], %[[DM]])
// CHECK-NEXT:    %[[M:.+]] = cir.call @malloc(%[[N]])
// CHECK-NEXT:    cir.store %[[N]], %[[SIZE]]
// CHECK-NEXT:    %[[OLD:.+]] = cir.load %[[SIZE]]
// CHECK-NEXT:    %[[LT:.+]] = cir.cmp lt %[[OLD]], %[[N2:[^ ]+]] : !u64i
// CHECK-NEXT:    %[[K:.+]] = cir.select if %[[LT]] then %[[OLD]] else %[[N2]]
// CHECK-NEXT:    %[[DR:.+]] = cir.call @realloc(%{{.+}}, %[[N2]])
// CHECK-NEXT:    cir.libc.memset %[[N2]] bytes at %[[DR]]
// CHECK-NEXT:    "enzyme.push"(%[[DRCACHE:.+]], %[[DR]])
// CHECK-NEXT:    "enzyme.push"(%[[OLDCACHE:.+]], %[[DM]])
// CHECK-NEXT:    "enzyme.push"(%[[KCACHE:.+]], %[[K]])
// CHECK-NEXT:    %[[R:.+]] = cir.call @realloc(%[[M]], %[[N2]])
// CHECK:         cir.call @free(%[[R]])
// CHECK:         %[[DRP:.+]] = "enzyme.pop"(%[[DRCACHE]])
// CHECK-NEXT:    %[[OLDP:.+]] = "enzyme.pop"(%[[OLDCACHE]])
// CHECK-NEXT:    "enzyme.pop"(%[[KCACHE]])
// CHECK:         cir.for
// CHECK:           cir.fadd
// CHECK:         cir.call @free(%[[DRP]])
// CHECK-NEXT:    %[[DMP:.+]] = "enzyme.pop"(%[[DMCACHE]])
// CHECK-NEXT:    cir.call @free(%[[DMP]])
