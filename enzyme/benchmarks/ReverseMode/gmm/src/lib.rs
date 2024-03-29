#![feature(autodiff)]
use libm::lgamma;

#[no_mangle]
pub extern "C" fn rust_dgmm_objective(d: i32, k: i32, n: i32, alphas: *const f64, dalphas: *mut f64, means: *const f64, dmeans: *mut f64, icf: *const f64, dicf: *mut f64, x: *const f64, wishart: *const Wishart, err: *mut f64, derr: *mut f64) {
    //dgmm_objective(d as usize, k as usize, n as usize, alphas, dalphas, means, dmeans, icf, dicf, x, wishart, err, derr);
}

#[autodiff(dgmm_objective, Reverse, Const, Const, Const, Duplicated, Duplicated, Duplicated, Const, Const, Duplicated)]
pub fn gmm_objective_c(d: usize, k: usize, n: usize, alphas: *const f64, means: *const f64, icf: *const f64, x: *const f64, wishart: *const Wishart, err: *mut f64) {
    let alphas = unsafe { std::slice::from_raw_parts(alphas, k) };
    let means = unsafe { std::slice::from_raw_parts(means, k * d) };
    let icf = unsafe { std::slice::from_raw_parts(icf, k * d * (d + 1) / 2) };
    let x = unsafe { std::slice::from_raw_parts(x, n * d) };
    let wishart: Wishart = unsafe { *wishart };
    let mut err = unsafe { *err };
    gmm_objective(d, k, n, alphas, means, icf, x, wishart, &mut err);
}

pub fn gmm_objective(d: usize, k: usize, n: usize, alphas: &[f64], means: &[f64], icf: &[f64], x: &[f64], wishart: Wishart, err: &mut f64) {
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

fn arr_max(n: usize, x: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for i in 0..n {
        if max < x[i] {
            max = x[i];
        }
    }
    max
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
    pub gamma: f64,
    pub m: i32,
}
fn log_wishart_prior(p: usize, k: usize, wishart: Wishart, sum_qs: &[f64], qdiags: &[f64], icf: &[f64]) -> f64 {
    let n = p + wishart.m as usize + 1;
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
