#![feature(autodiff)]

fn main() {
    let cam = [0.0; 9];
    let mut dcam = [0.0; 9];
    let x = [0.0; 3];
    let mut dx = [0.0; 3];
    let w = [0.0; 1];
    let mut dw = [0.0; 1];
    let feat = [0.0; 2];
    let mut err = [0.0; 2];
    let mut derr = [0.0; 2];
    dcompute_reproj_error(
        &cam as *const f64,
        &mut dcam as *mut f64,
        &x as *const f64,
        &mut dx as *mut f64,
        &w as *const f64,
        &mut dw as *mut f64,
        &feat as *const f64,
        &mut err as *mut f64,
        &mut derr as *mut f64,
    );
    let mut wb = 0.0;
    //compute zach 
    //dcompute_zach_weight_error(&w, &mut dw, &mut err, &mut derr);
    dcompute_zach_weight_error(&w as *const f64, &mut dw as *mut f64, &mut err as *mut f64, &mut derr as *mut f64);
}

fn sqsum(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

fn cross(a: &[f64], b: &[f64], out: &mut [f64]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

fn radial_distort(rad_params: &[f64], proj: &mut [f64]) {
    let rsq = sqsum(proj);
    let l = 1. + rad_params[0] * rsq + rad_params[1] * rsq * rsq;
    proj[0] = proj[0] * l;
    proj[1] = proj[1] * l;
}

fn rodrigues_rotate_point(rot: &[f64], pt: &[f64], rotated_pt: &mut [f64]) {
    let sqtheta = sqsum(rot);
    if sqtheta != 0. {
        let theta = sqtheta.sqrt();
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let theta_inverse = 1. / theta;
        let w = rot.iter().map(|&v| v * theta_inverse).collect::<Vec<_>>();
        let mut w_cross_pt = [0.; 3];
        cross(&w, &pt, &mut w_cross_pt);
        let tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (1. - costheta);
        for i in 0..3 {
            rotated_pt[i] = pt[i] * costheta + w_cross_pt[i] * sintheta + w[i] * tmp;
        }
    } else {
        let mut rot_cross_pt = [0.; 3];
        cross(&rot, &pt, &mut rot_cross_pt);
        for i in 0..3 {
            rotated_pt[i] = pt[i] + rot_cross_pt[i];
        }
    }
}

fn project(cam: &[f64], x: &[f64], proj: &mut [f64]) {
    let c = &cam[3..6];
    let mut xo = [0.; 3];
    let mut xcam = [0.; 3];

    for i in 0..3 {
        xo[i] = x[i] - c[i];
    }

    rodrigues_rotate_point(&cam[0..3], &xo, &mut xcam);

    proj[0] = xcam[0] / xcam[2];
    proj[1] = xcam[1] / xcam[2];

    radial_distort(&cam[9..], proj);

    proj[0] = proj[0] * cam[6] + cam[7];
    proj[1] = proj[1] * cam[6] + cam[8];
}

//dgmm_objective(d, k, n, alphas.as_ptr(), d_alphas.as_mut_ptr(), means.as_ptr(), d_means.as_mut_ptr(), icf.as_ptr(), d_icf.as_mut_ptr(), x.as_ptr(), wishart2 as *const Wishart, err2 as *mut f64, d_err2 as *mut f64);
#[autodiff(dcompute_reproj_error, Reverse, Duplicated, Duplicated, Duplicated, Const, Duplicated)]
pub fn compute_reproj_error(cam: *const f64, x: *const f64, w: *const f64, feat: *const f64, err: *mut f64) {
    let cam = unsafe { std::slice::from_raw_parts(cam, 9) };
    let x = unsafe { std::slice::from_raw_parts(x, 3) };
    let w = unsafe { std::slice::from_raw_parts(w, 1) };
    let feat = unsafe { std::slice::from_raw_parts(feat, 2) };
    let mut err = unsafe { std::slice::from_raw_parts_mut(err, 2) };
    let mut proj = [0.; 2];
    project(cam, x, &mut proj);
    err[0] = w[0] * (proj[0] - feat[0]);
    err[1] = w[0] * (proj[1] - feat[1]);
}

#[autodiff(dcompute_zach_weight_error, Reverse, Duplicated, Duplicated)]
pub fn compute_zach_weight_error(w: *const f64, err: *mut f64) {
    let w = unsafe { std::slice::from_raw_parts(w, 1) };
    let mut err = unsafe { std::slice::from_raw_parts_mut(err, 1) };
    err[0] = 1. - w[0] * w[0];
}

fn ba_objective(
    n: usize,
    m: usize,
    p: usize,
    cams: &[f64],
    x: &[f64],
    w: &[f64],
    obs: &[i32],
    feats: &[f64],
    reproj_err: &mut [f64],
    w_err: &mut [f64],
) {
    for i in 0..p {
        let cam_idx = obs[i * 2 + 0] as usize;
        let pt_idx = obs[i * 2 + 1] as usize;
        compute_reproj_error(
            cams[cam_idx * 9..].as_ptr(),
            x[pt_idx * 3..].as_ptr(),
            w[i..].as_ptr(),
            feats[i * 2..].as_ptr(),
             reproj_err[i * 2..].as_mut_ptr(),
        );
    }

    for i in 0..p {
        compute_zach_weight_error(w[i..].as_ptr(), w_err[i..].as_mut_ptr());
    }
}

//void compute_reproj_error_b(const double *cam, double *camb, const double *X,
//        double *Xb, const double *w, double *wb, const double *feat, double *
//        err, double *errb) {
//}
//void compute_zach_weight_error_b(const double *w, double *wb, double *err,
//        double *errb) {
//}

//extern "C" {
//    void ba_objective(
//        int n,
//        int m,
//        int p,
//        double const* cams,
//        double const* X,
//        double const* w,
//        int const* obs,
//        double const* feats,
//        double* reproj_err,
//        double* w_err
//    );
//
//    void dcompute_reproj_error(
//        double const* cam,
//        double * dcam,
//        double const* X,
//        double * dX,
//        double const* w,
//        double * wb,
//        double const* feat,
//        double *err,
//        double *derr
//    );
//
//    void dcompute_zach_weight_error(double const* w, double* dw, double* err, double* derr);
//
//    void compute_reproj_error_b(
//        double const* cam,
//        double * dcam,
//        double const* X,
//        double * dX,
//        double const* w,
//        double * wb,
//        double const* feat,
//        double *err,
//        double *derr
//    );
//
//    void compute_zach_weight_error_b(double const* w, double* dw, double* err, double* derr);
//}
