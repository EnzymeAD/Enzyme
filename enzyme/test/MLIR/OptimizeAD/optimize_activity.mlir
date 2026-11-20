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

  func.func @rev_used_active_arg(%x: f32, %y: f32, %dp: f32)
      -> (f32, f32, f32) {
    %p, %dx, %dy = enzyme.autodiff @square2(%x, %y, %dp)
        <{activity = [#enzyme.activity<enzyme_active>,
                     #enzyme.activity<enzyme_active>],
         ret_activity = [#enzyme.activity<enzyme_active>,
                         #enzyme.activity<enzyme_constnoneed>]}>
        : (f32, f32, f32) -> (f32, f32, f32)
    return %p, %dx, %dy : f32, f32, f32
  }
}

// CLASSIC-LABEL: func.func @rev_return_activity
// CLASSIC: enzyme.autodiff @square2(%arg0, %arg1, %arg2) {{.*}}activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]{{.*}}ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]

// DATAFLOW-LABEL: func.func @rev_return_activity
// DATAFLOW: enzyme.autodiff @square2(%arg0, %arg1, %arg2) {{.*}}activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>]{{.*}}ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]

// DATAFLOW-LABEL: func.func @rev_used_active_arg
// DATAFLOW-DAG: %[[R:.*]]:2 = enzyme.autodiff @square2(%arg0, %arg1, %arg2) {{.*}}activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_constnoneed>]{{.*}} : (f32, f32, f32) -> (f32, f32)
// DATAFLOW-DAG: %[[RZERO:.*]] = arith.constant 0.000000e+00 : f32
// DATAFLOW: return %[[R]]#0, %[[R]]#1, %[[RZERO]] : f32, f32, f32

// -----

module {
  func.func @identity(%c: f64) -> f64 {
    return %c : f64
  }
  func.func @fwd_used_dupnoneed(%c: f64) -> f64 {
    %dc = enzyme.fwddiff @identity(%c) <{
      activity = [#enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>]
    }> : (f64) -> f64
    return %dc : f64
  }
  func.func @fwd_used_dup(%c: f64) -> (f64, f64) {
    %p, %dc = enzyme.fwddiff @identity(%c) <{
      activity = [#enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (f64) -> (f64, f64)
    return %p, %dc : f64, f64
  }
}

// CLASSIC-LABEL: func.func @fwd_used_dupnoneed
// CLASSIC-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CLASSIC-DAG: enzyme.fwddiff @identity(%arg0) {{.*}}activity = [#enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_constnoneed>]{{.*}} : (f64) -> ()
// CLASSIC: return %[[ZERO]] : f64

// CLASSIC-LABEL: func.func @fwd_used_dup
// CLASSIC-DAG: %[[P:.*]] = enzyme.fwddiff @identity(%arg0) {{.*}}activity = [#enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_const>]{{.*}} : (f64) -> f64
// CLASSIC-DAG: %[[ZERO2:.*]] = arith.constant 0.000000e+00 : f64
// CLASSIC: return %[[P]], %[[ZERO2]] : f64, f64

// DATAFLOW-LABEL: func.func @fwd_used_dupnoneed
// DATAFLOW-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// DATAFLOW-DAG: enzyme.fwddiff @identity(%arg0) {{.*}}activity = [#enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_constnoneed>]{{.*}} : (f64) -> ()
// DATAFLOW: return %[[ZERO]] : f64

// DATAFLOW-LABEL: func.func @fwd_used_dup
// DATAFLOW-DAG: %[[P:.*]] = enzyme.fwddiff @identity(%arg0) {{.*}}activity = [#enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_const>]{{.*}} : (f64) -> f64
// DATAFLOW-DAG: %[[ZERO2:.*]] = arith.constant 0.000000e+00 : f64
// DATAFLOW: return %[[P]], %[[ZERO2]] : f64, f64

