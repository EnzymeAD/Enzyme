#![feature(autodiff)]
#![crate_type = "dylib"]
use libm::lgamma;

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
    dgmm_objective(d, k, n, alphas.as_ptr(), d_alphas.as_mut_ptr(), means.as_ptr(), d_means.as_mut_ptr(), icf.as_ptr(), d_icf.as_mut_ptr(), x.as_ptr(), wishart2 as *const Wishart, err2 as *mut f64, d_err2 as *mut f64);
}
fn arr_max(n: usize, x: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for i in 0..n {
        if max < x[i] {
            max = x[i];
        }
    }
    max
}

#[no_mangle]
pub extern "C" fn dgmm_objective_C(d: usize, k: usize, n: usize, alphas: *const f64, dalphas: *mut f64, means: *const f64, dmeans: *mut f64, icf: *const f64, dicf: *mut f64, x: *const f64, wishart: *const Wishart, err: *mut f64, derr: *mut f64) {
    dgmm_objective(d, k, n, alphas, dalphas, means, dmeans, icf, dicf, x, wishart, err, derr);
}

#[autodiff(dgmm_objective, Reverse, Const, Const, Const, Duplicated, Duplicated, Duplicated, Const, Const, Duplicated)]
fn gmm_objective_c(d: usize, k: usize, n: usize, alphas: *const f64, means: *const f64, icf: *const f64, x: *const f64, wishart: *const Wishart, err: *mut f64) {
    let alphas = unsafe { std::slice::from_raw_parts(alphas, k) };
    let means = unsafe { std::slice::from_raw_parts(means, k * d) };
    let icf = unsafe { std::slice::from_raw_parts(icf, k * d * (d + 1) / 2) };
    let x = unsafe { std::slice::from_raw_parts(x, n * d) };
    let wishart: Wishart = unsafe { *wishart };
    let mut err = unsafe { *err };
    gmm_objective(d, k, n, alphas, means, icf, x, wishart, &mut err);
}

fn gmm_objective(d: usize, k: usize, n: usize, alphas: &[f64], means: &[f64], icf: &[f64], x: &[f64], wishart: Wishart, err: &mut f64) {
    let constant = -(n as f64) * d as f64 * 0.5 * 2f64.ln();
    let icf_sz = d * (d + 1) / 2;
    let mut qdiags = vec![0.; d * k];
    let mut sum_qs = vec![0.; k];
    let mut xcentered = vec![0.; d];
    let mut qxcentered = vec![0.; d];
    let mut main_term = vec![0.; k];

    preprocess_qs(d, k, icf, &mut sum_qs, &mut qdiags);

    let mut slse = 0.;
    for ix in 0..n {
        for ik in 0..k {
            subtract(d, &x[ix as usize * d as usize..], &means[ik as usize * d as usize..], &mut xcentered);
            Qtimesx(d, &qdiags[ik as usize * d as usize..], &icf[ik as usize * icf_sz as usize + d as usize..], &xcentered, &mut qxcentered);
            main_term[ik as usize] = alphas[ik as usize] + sum_qs[ik as usize] - 0.5 * sqnorm(d, &qxcentered);
        }

        slse = slse + log_sum_exp(k, &main_term);
    }

    let lse_alphas = log_sum_exp(k, alphas);

    *err = constant + slse - n as f64 * lse_alphas + log_wishart_prior(d, k, wishart, &sum_qs, &qdiags, icf);
}

fn preprocess_qs(d: usize, k: usize, icf: &[f64], sum_qs: &mut [f64], qdiags: &mut [f64]) {
    let icf_sz = d * (d + 1) / 2;
    for ik in 0..k {
        sum_qs[ik as usize] = 0.;
        for id in 0..d {
            let q = icf[ik as usize * icf_sz as usize + id as usize];
            sum_qs[ik as usize] = sum_qs[ik as usize] + q;
            qdiags[ik as usize * d as usize + id as usize] = q.exp();
        }
    }
}
fn subtract(d: usize, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() >= d);
    assert!(y.len() >= d);
    assert!(out.len() >= d);
    for i in 0..d {
        out[i] = x[i] - y[i];
    }
}

fn Qtimesx(d: usize, q_diag: &[f64], ltri: &[f64], x: &[f64], out: &mut [f64]) {
    assert!(out.len() >= d);
    assert!(q_diag.len() >= d);
    assert!(x.len() >= d);
    for i in 0..d {
        out[i] = q_diag[i] * x[i];
    }

    for i in 0..d {
        let mut lparamsidx = i*(2*d-i-1)/2;
        for j in i + 1..d {
            out[j] = out[j] + ltri[lparamsidx] * x[i];
            lparamsidx += 1;
        }
    }
}

fn log_sum_exp(n: usize, x: &[f64]) -> f64 {
    let mx = arr_max(n, x);
    let semx: f64 = x.iter().map(|x| (x - mx).exp()).sum();
    semx.ln() + mx
}
fn log_gamma_distrib(a: f64, p: f64) -> f64 {
    0.25 * p * (p - 1.) * std::f64::consts::PI.ln() + (1..=p as usize).map(|j| lgamma(a + 0.5 * (1. - j as f64))).sum::<f64>()
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Wishart {
    gamma: f64,
    m: usize,
}
fn log_wishart_prior(p: usize, k: usize, wishart: Wishart, sum_qs: &[f64], qdiags: &[f64], icf: &[f64]) -> f64 {
    let n = p + wishart.m + 1;
    let icf_sz = p * (p + 1) / 2;

    let c = n as f64 * p as f64 * (wishart.gamma.ln() - 0.5 * 2f64.ln()) - log_gamma_distrib(0.5 * n as f64, p as f64);

    let out = (0..k).map(|ik| {
        let frobenius = sqnorm(p, &qdiags[ik * p as usize..]) + sqnorm(icf_sz - p, &icf[ik * icf_sz as usize + p as usize..]);
        0.5 * wishart.gamma * wishart.gamma * (frobenius) - (wishart.m as f64) * sum_qs[ik as usize]
    }).sum::<f64>();

    out - k as f64 * c
}

fn sqnorm(n: usize, x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum()
}
