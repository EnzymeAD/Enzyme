// RUN: %eopt --enzyme-inline-callees %s | FileCheck %s

// The calls in what enzyme.autodiff differentiates, @f, are inlined, and so
// are those in what is inlined: @f is differentiated as one function, with
// no derivative of each helper and no tape between them. A callee's return
// becomes a branch to what follows the call, and its stack slots move to the
// entry block. A recursive callee, an indirect call, and the functions that
// are not differentiated, as @g, are left alone.

!fnty = !cir.func<(!cir.double) -> !cir.double>
module {
  cir.func @sq(%x: !cir.double) -> !cir.double {
    %t = cir.alloca "t" align(8) : !cir.ptr<!cir.double>
    %m = cir.fmul %x, %x : !cir.double
    cir.store %m, %t : !cir.double, !cir.ptr<!cir.double>
    %r = cir.load %t : !cir.ptr<!cir.double>, !cir.double
    cir.return %r : !cir.double
  }
  cir.func @pick(%x: !cir.double, %c: !cir.bool) -> !cir.double {
    cir.brcond %c ^small, ^big
  ^small:
    cir.return %x : !cir.double
  ^big:
    %y = cir.call @sq(%x) : (!cir.double) -> !cir.double
    cir.return %y : !cir.double
  }
  cir.func @rec(%x: !cir.double) -> !cir.double {
    %r = cir.call @rec(%x) : (!cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @f(%x: !cir.double, %c: !cir.bool, %fp: !cir.ptr<!fnty>) -> !cir.double {
    %a = cir.call @pick(%x, %c) : (!cir.double, !cir.bool) -> !cir.double
    %b = cir.call @rec(%a) : (!cir.double) -> !cir.double
    %r = cir.call %fp(%b) : (!cir.ptr<!fnty>, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @g(%x: !cir.double) -> !cir.double {
    %r = cir.call @sq(%x) : (!cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
  cir.func @rev(%x: !cir.double, %c: !cir.bool, %fp: !cir.ptr<!fnty>, %seed: !cir.double) -> !cir.double {
    %r = enzyme.autodiff @f(%x, %c, %fp, %seed) <{ activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity=[#enzyme.activity<enzyme_activenoneed>] }> : (!cir.double, !cir.bool, !cir.ptr<!fnty>, !cir.double) -> !cir.double
    cir.return %r : !cir.double
  }
}

// CHECK-LABEL: cir.func @f(
// CHECK-SAME:    %[[X:[^:]+]]: !cir.double, %[[C:[^:]+]]: !cir.bool, %[[FP:[^:]+]]: !cir.ptr
// CHECK-NEXT:    %[[T:.+]] = cir.alloca "t" align(8) : !cir.ptr<!cir.double>
// CHECK-NEXT:    cir.brcond %[[C]] ^[[SMALL:.+]], ^[[BIG:[^ ]+]]
// CHECK:       ^[[SMALL]]:
// CHECK-NEXT:    cir.br ^[[AFTER:.+]](%[[X]] : !cir.double)
// CHECK:       ^[[BIG]]:
// CHECK-NEXT:    %[[M:.+]] = cir.fmul %[[X]], %[[X]]
// CHECK-NEXT:    cir.store %[[M]], %[[T]]
// CHECK-NEXT:    %[[R:.+]] = cir.load %[[T]]
// CHECK-NEXT:    cir.br ^[[AFTER]](%[[R]] : !cir.double)
// CHECK:       ^[[AFTER]](%[[A:.+]]: !cir.double):
// CHECK-NEXT:    %[[B:.+]] = cir.call @rec(%[[A]])
// CHECK-NEXT:    %[[I:.+]] = cir.call %[[FP]](%[[B]])
// CHECK-NEXT:    cir.return %[[I]]

// CHECK-LABEL: cir.func @g(
// CHECK-NEXT:    cir.call @sq(
