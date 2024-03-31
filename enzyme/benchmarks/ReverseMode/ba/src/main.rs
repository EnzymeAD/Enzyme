use bars::{dcompute_reproj_error, dcompute_zach_weight_error};
fn main() {
    let cam = [0.0; 11];
    let mut dcam = [0.0; 11];
    let x = [0.0; 3];
    let mut dx = [0.0; 3];
    let w = [0.0; 1];
    let mut dw = [0.0; 1];
    let feat = [0.0; 2];
    let mut err = [0.0; 2];
    let mut derr = [0.0; 2];
    dcompute_reproj_error(
        &cam as *const [f64;11],
        &mut dcam as *mut [f64;11],
        &x as *const [f64;3],
        &mut dx as *mut [f64;3],
        &w as *const [f64;1],
        &mut dw as *mut [f64;1],
        &feat as *const [f64;2],
        &mut err as *mut [f64;2],
        &mut derr as *mut [f64;2],
    );

    let mut wb = 0.0;
    dcompute_zach_weight_error(&w as *const f64, &mut dw as *mut f64, &mut err as *mut f64, &mut derr as *mut f64);
}
