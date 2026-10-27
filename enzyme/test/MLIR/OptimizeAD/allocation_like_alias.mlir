// RUN: %eopt --split-input-file --hoist-enzyme-regions %s | FileCheck %s

// A write to one fresh heap allocation cannot block hoisting a read from
// another. This exercises the MallocLike query in oputils::mayAlias.
// CHECK-LABEL: func.func @independent_heap_allocations(
// CHECK: %[[A:.*]] = memref.alloc()
// CHECK: %[[B:.*]] = memref.alloc()
// CHECK: memref.store %{{.*}}, %[[B]][]
// CHECK: %[[LOAD:.*]] = memref.load %[[B]][]
// CHECK: enzyme.autodiff_region
// CHECK: memref.store %{{.*}}, %[[A]][]
// CHECK-NOT: memref.load
// CHECK: arith.addf %{{.*}}, %[[LOAD]]
// CHECK: enzyme.yield
func.func @independent_heap_allocations(%x: f64, %dx: f64) -> f64 {
  %a = memref.alloc() : memref<f64>
  %b = memref.alloc() : memref<f64>
  %one = arith.constant 1.0 : f64
  memref.store %one, %b[] : memref<f64>
  %result = enzyme.autodiff_region(%x, %dx) {
  ^bb0(%active: f64):
    memref.store %active, %a[] : memref<f64>
    %value = memref.load %b[] : memref<f64>
    %sum = arith.addf %active, %value : f64
    enzyme.yield %sum : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  memref.dealloc %a : memref<f64>
  memref.dealloc %b : memref<f64>
  return %result : f64
}

// -----

// A read of the buffer just written must stay inside the region.
// CHECK-LABEL: func.func @same_heap_allocation(
// CHECK: %[[BUFFER:.*]] = memref.alloc()
// CHECK-NOT: memref.load
// CHECK: enzyme.autodiff_region
// CHECK: memref.store %{{.*}}, %[[BUFFER]][]
// CHECK-NEXT: %[[LOAD:.*]] = memref.load %[[BUFFER]][]
// CHECK-NEXT: enzyme.yield %[[LOAD]]
func.func @same_heap_allocation(%x: f64, %dx: f64) -> f64 {
  %buffer = memref.alloc() : memref<f64>
  %result = enzyme.autodiff_region(%x, %dx) {
  ^bb0(%active: f64):
    memref.store %active, %buffer[] : memref<f64>
    %value = memref.load %buffer[] : memref<f64>
    enzyme.yield %value : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  memref.dealloc %buffer : memref<f64>
  return %result : f64
}

// -----

// An affine load does not capture its base allocation. Even an unknown call
// result cannot alias this fresh, non-captured buffer. This also exercises the
// LoadLike query in isCaptured after removing the explicit AffineLoadOp check.
// CHECK-LABEL: func.func @affine_load_does_not_capture(
// CHECK: %[[BUFFER:.*]] = memref.alloc()
// CHECK: %[[OTHER:.*]] = {{.*}}call @get_buffer
// CHECK: affine.store %{{.*}}, %[[BUFFER]][0]
// CHECK: %[[LOAD:.*]] = affine.load %[[BUFFER]][0]
// CHECK: enzyme.autodiff_region
// CHECK: memref.store %{{.*}}, %[[OTHER]][%{{.*}}]
// CHECK-NOT: affine.load
// CHECK: arith.addf %{{.*}}, %[[LOAD]]
// CHECK: enzyme.yield
func.func private @get_buffer() -> memref<1xf64>
func.func @affine_load_does_not_capture(%x: f64, %dx: f64) -> f64 {
  %buffer = memref.alloc() : memref<1xf64>
  %other = func.call @get_buffer() : () -> memref<1xf64>
  %one = arith.constant 1.0 : f64
  %zero = arith.constant 0 : index
  affine.store %one, %buffer[0] : memref<1xf64>
  %result = enzyme.autodiff_region(%x, %dx) {
  ^bb0(%active: f64):
    memref.store %active, %other[%zero] : memref<1xf64>
    %value = affine.load %buffer[0] : memref<1xf64>
    %sum = arith.addf %active, %value : f64
    enzyme.yield %sum : f64
  } <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (f64, f64) -> f64
  memref.dealloc %buffer : memref<1xf64>
  return %result : f64
}
