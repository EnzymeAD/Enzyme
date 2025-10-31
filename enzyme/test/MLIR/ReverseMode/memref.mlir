// RUN: %eopt --split-input-file --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math %s | FileCheck %s

func.func @subview(%mem: memref<4x3xf32, strided<[?, ?], offset: ?>>, %x: index, %y: index) -> f32 {
  %row = memref.subview %mem[%x, 0] [1, 3] [1, 1] : memref<4x3xf32, strided<[?, ?], offset: ?>> to memref<3xf32, strided<[?], offset: ?>>
  %val = memref.load %row[%y] : memref<3xf32, strided<[?], offset: ?>>
  return %val : f32
}

func.func @dsubview(
  %mem: memref<4x3xf32, strided<[?, ?], offset: ?>>, 
  %dmem: memref<4x3xf32, strided<[?, ?], offset: ?>>,
  %x: index, %y: index, %dout: f32
) {
  enzyme.autodiff @subview(%mem, %dmem, %x, %y, %dout)
    {
      activity=[
        #enzyme<activity enzyme_dup>,
        #enzyme<activity enzyme_const>,
        #enzyme<activity enzyme_const>
      ],
      ret_activity=[#enzyme<activity enzyme_activenoneed>]
    } : (
      memref<4x3xf32, strided<[?, ?], offset: ?>>,
      memref<4x3xf32, strided<[?, ?], offset: ?>>,
      index, index, f32
    ) -> ()
  return
}

// CHECK: func.func private @diffesubview(%arg0: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg2: index, %arg3: index, %arg4: f32) {
// CHECK-NEXT:    %subview = memref.subview %arg1[%arg2, 0] [1, 3] [1, 1] : memref<4x3xf32, strided<[?, ?], offset: ?>> to memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:    %0 = memref.load %subview[%arg3] : memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:    %1 = arith.addf %0, %arg4 : f32
// CHECK-NEXT:    memref.store %1, %subview[%arg3] : memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

func.func @subview_in_loop(%mem: memref<4x3xf32, strided<[?, ?], offset: ?>>, %y: index, %out: memref<f32>) {
  affine.for %iv = 0 to 4 {
    %row = memref.subview %mem[%iv, 0] [1, 3] [1, 1] : memref<4x3xf32, strided<[?, ?], offset: ?>> to memref<3xf32, strided<[?], offset: ?>>
    %val = memref.load %row[%y] : memref<3xf32, strided<[?], offset: ?>>
    %prev = memref.load %out[] : memref<f32>
    %next = arith.addf %val, %prev : f32
    memref.store %next, %out[] : memref<f32>
  }
  return
}

func.func @dsubview(
  %mem: memref<4x3xf32, strided<[?, ?], offset: ?>>,
  %dmem: memref<4x3xf32, strided<[?, ?], offset: ?>>,
  %y: index, %out: memref<f32>, %dout: memref<f32>
) {
  enzyme.autodiff @subview_in_loop(%mem, %dmem, %y, %out, %dout)
    {
      activity=[
        #enzyme<activity enzyme_dup>,
        #enzyme<activity enzyme_const>,
        #enzyme<activity enzyme_dupnoneed>
      ],
      ret_activity=[]
    } : (
      memref<4x3xf32, strided<[?, ?], offset: ?>>,
      memref<4x3xf32, strided<[?, ?], offset: ?>>,
      index, memref<f32>, memref<f32>
    ) -> ()
  return
}

// CHECK: func.func private @diffesubview_in_loop(%arg0: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg1: memref<4x3xf32, strided<[?, ?], offset: ?>>, %arg2: index, %arg3: memref<f32>, %arg4: memref<f32>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    affine.for %arg5 = 0 to 4 {
// CHECK-NEXT:      %subview = memref.subview %arg0[%arg5, 0] [1, 3] [1, 1] : memref<4x3xf32, strided<[?, ?], offset: ?>> to memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:      %0 = memref.load %subview[%arg2] : memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:      %1 = memref.load %arg3[] : memref<f32>
// CHECK-NEXT:      %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:      memref.store %2, %arg3[] : memref<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.for %arg5 = 0 to 4 {
// CHECK-NEXT:      %subview = memref.subview %arg1[%arg5, 0] [1, 3] [1, 1] : memref<4x3xf32, strided<[?, ?], offset: ?>> to memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:      %0 = memref.load %arg4[] : memref<f32>
// CHECK-NEXT:      memref.store %cst, %arg4[] : memref<f32>
// CHECK-NEXT:      %1 = memref.load %arg4[] : memref<f32>
// CHECK-NEXT:      %2 = arith.addf %1, %0 : f32
// CHECK-NEXT:      memref.store %2, %arg4[] : memref<f32>
// CHECK-NEXT:      %3 = memref.load %subview[%arg2] : memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:      %4 = arith.addf %3, %0 : f32
// CHECK-NEXT:      memref.store %4, %subview[%arg2] : memref<3xf32, strided<[?], offset: ?>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
