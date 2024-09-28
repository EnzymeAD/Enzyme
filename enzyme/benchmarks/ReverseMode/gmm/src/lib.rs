#![feature(autodiff)]
pub mod r#unsafe;
pub mod safe;

use r#unsafe::dgmm_objective as dgmm_objective;


#[derive(Clone, Copy)]
#[repr(C)]
pub struct Wishart {
    pub gamma: f64,
    pub m: i32,
}

