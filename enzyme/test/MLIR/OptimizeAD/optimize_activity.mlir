// RUN: %eopt --enzyme-activity-opt --split-input-file %s | FileCheck %s --check-prefix=CLASSIC
// RUN: %eopt --enzyme-activity-opt="dataflow=true" --split-input-file %s | FileCheck %s --check-prefix=DATAFLOW

module {
  func.func @square2(%x: f32 {enzyme.tag = "x"},
                     %y: f32 {enzyme.tag = "y"}) -> (f32, f32) {
    %p = arith.mulf %x, %x {tag = "p"} : f32
    %q = arith.mulf %y, %y {tag = "q"} : f32
    return %p, %q : f32, f32
  }

  func.func @fwd_return_activity(%x: f32, %y: f32, %dy: f32) -> f32 {
    %p, %dp = enzyme.fwddiff @square2(%x, %y, %dy)
        <{activity = [#enzyme.activity<enzyme_const>,
                     #enzyme.activity<enzyme_dup>],
         ret_activity = [#enzyme.activity<enzyme_dup>,
                         #enzyme.activity<enzyme_constnoneed>]}>
        : (f32, f32, f32) -> (f32, f32)
    return %p : f32
  }

  func.func @fwd_argument_activity(%x: f32, %y: f32, %dx: f32, %dy: f32)
      -> (f32, f32, f32) {
    %p, %q, %dq = enzyme.fwddiff @square2(%x, %dx, %y, %dy)
        <{activity = [#enzyme.activity<enzyme_dup>,
                     #enzyme.activity<enzyme_dup>],
         ret_activity = [#enzyme.activity<enzyme_const>,
                         #enzyme.activity<enzyme_dup>]}>
        : (f32, f32, f32, f32) -> (f32, f32, f32)
    return %p, %q, %dq : f32, f32, f32
  }
}

// CLASSIC-LABEL: func.func @fwd_return_activity
// CLASSIC: enzyme.fwddiff @square2(%arg0, %arg1, %arg2) {{.*}}activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>]{{.*}}ret_activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_constnoneed>]

// CLASSIC-LABEL: func.func @fwd_argument_activity
// CLASSIC: enzyme.fwddiff @square2(%arg0, %arg2, %arg1, %arg3) {{.*}}activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]{{.*}}ret_activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>]

// DATAFLOW-LABEL: func.func @fwd_return_activity
// DATAFLOW: enzyme.fwddiff @square2(%arg0, %arg1) {{.*}}activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>]{{.*}}ret_activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_constnoneed>]

// DATAFLOW-LABEL: func.func @fwd_argument_activity
// DATAFLOW: enzyme.fwddiff @square2(%arg0, %arg1, %arg3) {{.*}}activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>]{{.*}}ret_activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>]

// -----

module {
  func.func @square2(%x: f32 {enzyme.tag = "x"},
                     %y: f32 {enzyme.tag = "y"}) -> (f32, f32) {
    %p = arith.mulf %x, %x {tag = "p"} : f32
    %q = arith.mulf %y, %y {tag = "q"} : f32
    return %p, %q : f32, f32
  }

  func.func @rev_return_activity(%x: f32, %y: f32, %dp: f32, %dq: f32)
      -> (f32, f32) {
    %p, %dx = enzyme.autodiff @square2(%x, %y, %dp, %dq)
        <{activity = [#enzyme.activity<enzyme_active>,
                     #enzyme.activity<enzyme_const>],
         ret_activity = [#enzyme.activity<enzyme_active>,
                         #enzyme.activity<enzyme_activenoneed>]}>
        : (f32, f32, f32, f32) -> (f32, f32)
    return %p, %dx : f32, f32
  }
}

// CLASSIC-LABEL: func.func @rev_return_activity
// CLASSIC: enzyme.autodiff @square2(%arg0, %arg1, %arg2) {{.*}}activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]{{.*}}ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]

// DATAFLOW-LABEL: func.func @rev_return_activity
// DATAFLOW: enzyme.autodiff @square2(%arg0, %arg1, %arg2) {{.*}}activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]{{.*}}ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]

// -----

module {
  func.func @struct_float(%a: !llvm.struct<(f64, f64)>,
                          %b: !llvm.struct<(f64, f64)>)
      -> !llvm.struct<(f64, f64)> {
    %b0 = llvm.extractvalue %b[0] : !llvm.struct<(f64, f64)>
    %sq = arith.mulf %b0, %b0 : f64
    %out = llvm.insertvalue %sq, %b[0] : !llvm.struct<(f64, f64)>
    return %out : !llvm.struct<(f64, f64)>
  }

  func.func @struct_ptr(%cap: !llvm.struct<(ptr, ptr)>) {
    %px = llvm.extractvalue %cap[0] : !llvm.struct<(ptr, ptr)>
    %py = llvm.extractvalue %cap[1] : !llvm.struct<(ptr, ptr)>
    %v = llvm.load %px : !llvm.ptr -> f64
    %sq = arith.mulf %v, %v : f64
    llvm.store %sq, %py : f64, !llvm.ptr
    return
  }

  func.func @fwd_struct_activity(
      %a: !llvm.struct<(f64, f64)>, %da: !llvm.struct<(f64, f64)>,
      %b: !llvm.struct<(f64, f64)>, %db: !llvm.struct<(f64, f64)>,
      %cap: !llvm.struct<(ptr, ptr)>, %dcap: !llvm.struct<(ptr, ptr)>)
      -> (!llvm.struct<(f64, f64)>, !llvm.struct<(f64, f64)>) {
    %r, %dr = enzyme.fwddiff @struct_float(%a, %da, %b, %db)
        <{activity = [#enzyme.activity<enzyme_dup>,
                     #enzyme.activity<enzyme_dup>],
         ret_activity = [#enzyme.activity<enzyme_dup>]}>
        : (!llvm.struct<(f64, f64)>, !llvm.struct<(f64, f64)>,
           !llvm.struct<(f64, f64)>, !llvm.struct<(f64, f64)>)
        -> (!llvm.struct<(f64, f64)>, !llvm.struct<(f64, f64)>)
    enzyme.fwddiff @struct_ptr(%cap, %dcap)
        <{activity = [#enzyme.activity<enzyme_dup>],
         ret_activity = []}>
        : (!llvm.struct<(ptr, ptr)>, !llvm.struct<(ptr, ptr)>) -> ()
    return %r, %dr : !llvm.struct<(f64, f64)>, !llvm.struct<(f64, f64)>
  }
}

// CLASSIC-LABEL: func.func @fwd_struct_activity
// CLASSIC: enzyme.fwddiff @struct_float(%arg0, %arg1, %arg2, %arg3) {{.*}}activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]{{.*}}ret_activity = [#enzyme.activity<enzyme_dup>]
// CLASSIC: enzyme.fwddiff @struct_ptr(%arg4, %arg5) {{.*}}activity = [#enzyme.activity<enzyme_dup>]{{.*}}ret_activity = []

// DATAFLOW-LABEL: func.func @fwd_struct_activity
// DATAFLOW: enzyme.fwddiff @struct_float(%arg0, %arg2, %arg3) {{.*}}activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>]{{.*}}ret_activity = [#enzyme.activity<enzyme_dup>]
// DATAFLOW: enzyme.fwddiff @struct_ptr(%arg4, %arg5) {{.*}}activity = [#enzyme.activity<enzyme_dup>]{{.*}}ret_activity = []
