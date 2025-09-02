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
  
