#![feature(slice_swap_unchecked)]
#![feature(autodiff)]
#![feature(core_intrinsics)]
#![feature(slice_as_chunks)]

pub mod safe;
pub mod unsf;

trait F64Ext {
    fn fmuladd(&self, b: Self, c: Self) -> Self;
}

impl F64Ext for f64 {
    fn fmuladd(&self, b: Self, c: Self) -> Self {
        unsafe { std::intrinsics::fmuladdf64(*self, b, c) }
    }
}
