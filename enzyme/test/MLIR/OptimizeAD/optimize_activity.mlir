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
        {activity = [#enzyme<activity enzyme_const>,
                     #enzyme<activity enzyme_dup>],
         ret_activity = [#enzyme<activity enzyme_dup>,
                         #enzyme<activity enzyme_constnoneed>]}
        : (f32, f32, f32) -> (f32, f32)
    return %p : f32
  }

  func.func @fwd_argument_activity(%x: f32, %y: f32, %dx: f32, %dy: f32)
      -> (f32, f32, f32) {
    %p, %q, %dq = enzyme.fwddiff @square2(%x, %dx, %y, %dy)
        {activity = [#enzyme<activity enzyme_dup>,
                     #enzyme<activity enzyme_dup>],
         ret_activity = [#enzyme<activity enzyme_const>,
                         #enzyme<activity enzyme_dup>]}
        : (f32, f32, f32, f32) -> (f32, f32, f32)
    return %p, %q, %dq : f32, f32, f32
  }
}

// CLASSIC-LABEL: func.func @fwd_return_activity
// CLASSIC: enzyme.fwddiff @square2(%arg0, %arg1, %arg2) {{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>]{{.*}}ret_activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_constnoneed>]

// CLASSIC-LABEL: func.func @fwd_argument_activity
// CLASSIC: enzyme.fwddiff @square2(%arg0, %arg2, %arg1, %arg3) {{.*}}activity = [#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>]{{.*}}ret_activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>]

// DATAFLOW-LABEL: func.func @fwd_return_activity
// DATAFLOW: enzyme.fwddiff @square2(%arg0, %arg1) {{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>]{{.*}}ret_activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_constnoneed>]

// DATAFLOW-LABEL: func.func @fwd_argument_activity
// DATAFLOW: enzyme.fwddiff @square2(%arg0, %arg1, %arg3) {{.*}}activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>]{{.*}}ret_activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>]
