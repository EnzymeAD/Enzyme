#![feature(autodiff)]

#[cfg(not(any(feature = "safe", feature = "unsf")))]
compile_error!("Enable at least one of: features `safe` or `unsf`");

#[cfg(all(feature = "safe"))]
pub mod safe;

#[cfg(all(feature = "unsf"))]
pub mod r#unsafe;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Wishart {
    pub gamma: f64,
    pub m: i32,
}
