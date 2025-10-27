#![feature(autodiff)]
#![allow(non_snake_case)]

#[cfg(not(any(feature = "safe", feature = "unsf")))]
compile_error!("Enable at least one of: features `safe` or `unsf`");

#[cfg(all(feature = "safe"))]
pub mod safe;
#[cfg(all(feature = "unsf"))]
pub mod r#unsafe;

