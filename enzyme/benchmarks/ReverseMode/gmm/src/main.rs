#![feature(autodiff)]
use gmmrs::{Wishart, r#unsafe::dgmm_objective};

fn main() {
    let d = 2;
    let k = 2;
    let n = 2;
    let alphas = vec![0.5, 0.5];
    let means = vec![0., 0., 1., 1.];
    let icf = vec![1., 0., 1.];
    let x = vec![0., 0., 1., 1.];
    let wishart = Wishart { gamma: 1., m: 1 };
    let mut err = 0.;
    let mut d_alphas = vec![0.; alphas.len()];
    let mut d_means = vec![0.; means.len()];
    let mut d_icf = vec![0.; icf.len()];
    let mut d_x = vec![0.; x.len()];
    let mut d_err = 0.;
    let mut err2 = &mut err;
    let mut d_err2 = &mut d_err;
    let wishart2 = &wishart;
    // pass as raw ptr:
    unsafe {dgmm_objective(d, k, n, alphas.as_ptr(), d_alphas.as_mut_ptr(), means.as_ptr(), d_means.as_mut_ptr(), icf.as_ptr(), d_icf.as_mut_ptr(), x.as_ptr(), wishart2 as *const Wishart, err2 as *mut f64, d_err2 as *mut f64);}
}
