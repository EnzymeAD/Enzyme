#![feature(autodiff)]
pub mod safe;
pub mod r#unsafe;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Wishart {
    pub gamma: f64,
    pub m: i32,
}
