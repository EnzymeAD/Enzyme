#![feature(autodiff)]
#![allow(non_snake_case)]

use std::autodiff::autodiff;
pub mod safe;
pub mod r#unsafe;

static BA_NCAMPARAMS: usize = 11;

#[no_mangle]
pub extern "C" fn rust_dcompute_zach_weight_error(
    w: *const f64,
    dw: *mut f64,
    err: *mut f64,
    derr: *mut f64,
) {
    dcompute_zach_weight_error(w, dw, err, derr);
}

#[autodiff(dcompute_zach_weight_error, Reverse, Duplicated, Duplicated)]
pub fn compute_zach_weight_error(w: *const f64, err: *mut f64) {
    let w = unsafe { *w };
    unsafe { *err = 1. - w * w; }
}

