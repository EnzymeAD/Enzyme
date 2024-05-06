#![feature(autodiff)]
#![feature(iter_next_chunk)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

//#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
//#define BOOST_NO_EXCEPTIONS

const N: usize = 32;
const xmin: f64 = 0.;
const xmax: f64 = 1.;
const ymin: f64 = 0.;
const ymax: f64 = 1.;

#[inline(always)]
fn range(min: f64, max: f64, i: usize, N_var: usize) -> f64 {
    (max - min) / (N_var as f64 - 1.) * i as f64 + min
}
#[inline(always)]
fn get(x: &[f64], i: usize, j: usize) -> f64 {
    assert!(i > 0);
    assert!(j < N);
    x[N * i + j]
}

//#define RANGE(min, max, i, N) ((max-min)/(N-1)*i + min)
//#define GETnb(x, i, j) (x)[N*i+j]
//#define GET(x, i, j) GETnb(x, i, j)
// #define GET(x, i, j) ({ assert(i >=0); assert( j>=0); assert(j<N);
// assert(j<N); GETnb(x, i, j); })

fn brusselator_f(x: f64, y: f64, t: f64) -> f64 {
    let eq1 = (x - 0.3) * (x - 0.3) + (y - 0.6) * (y - 0.6) <= 0.1 * 0.1;
    let eq2 = t >= 1.1;
    if eq1 && eq2 {
        5.0
    } else {
        0.0
    }
}

fn init_brusselator(u: &mut [f64], v: &mut [f64]) {
    for i in 0..N {
        for j in 0..N {
            let x = range(xmin, xmax, i, N);
            let y = range(ymin, ymax, j, N);
            u[N * i + j] = 22.0 * y * (1.0 - y) * (y * (1.0 - y)).sqrt();
            v[N * i + j] = 27.0 * x * (1.0 - x) * (x * (1.0 - x)).sqrt();
        }
    }
}
//    __enzyme_autodiff<void>(brusselator_2d_loop,
//                            enzyme_dupnoneed, nullptr, dadjoint_inp.data(),
//                            enzyme_dupnoneed, nullptr, dadjoint_inp.data() + N * N,
//                            enzyme_dup, x.data(), dx.data(),
//                            enzyme_dup, x.data() + N * N, dx.data() + N * N,
//                            enzyme_dup, p, dp,
//                            enzyme_const, t);


#[autodiff(dbrusselator_2d_loop, Reverse, Duplicated, Duplicated, Duplicated, Duplicated, Duplicated, Const)]
fn brusselator_2d_loop(d_u: &mut [f64], d_v: &mut [f64], u: &[f64], v: &[f64], p: &[f64;3], t: f64) {
    let A = p[0];
    let B = p[1];
    let alpha = p[2];
    let dx = 1. / (N - 1) as f64;
    let alpha = alpha / (dx * dx);
    for i in 0..N {
        for j in 0..N {
            let x = range(xmin, xmax, i, N);
            let y = range(ymin, ymax, j, N);
            let ip1 = if i == N - 1 { i } else { i + 1 };
            let im1 = if i == 0 { i } else { i - 1 };
            let jp1 = if j == N - 1 { j } else { j + 1 };
            let jm1 = if j == 0 { j } else { j - 1 };
            let u2v = u[N * i + j] * u[N * i + j] * v[N * i + j];
            d_u[N * i + j] = alpha * (u[N * im1 + j] + u[N * ip1 + j] + u[N * i + jp1] + u[N * i + jm1] - 4. * u[N * i + j])
                + B + u2v - (A + 1.) * u[N * i + j] + brusselator_f(x, y, t);
            d_v[N * i + j] = alpha * (v[N * im1 + j] + v[N * ip1 + j] + v[N * i + jp1] + v[N * i + jm1] - 4. * v[N * i + j])
                + A * u[N * i + j] - u2v;
        }
    }
}

//__attribute__((noinline))
//void brusselator_2d_loop(double* __restrict du, double* __restrict dv, const double* __restrict u, const double* __restrict v, const double* __restrict p, double t) {
//  double A = p[0];
//  double B = p[1];
//  double alpha = p[2];
//  double dx = (double)1/(N-1);
//
//  alpha = alpha/(dx*dx);
//
//  for(int i=0; i<N; i++) {
//    for(int j=0; j<N; j++) {
//
//      double x = RANGE(xmin, xmax, i, N);
//      double y = RANGE(ymin, ymax, j, N);
//
//      unsigned ip1 = (i == N-1) ? i : (i+1);
//      unsigned im1 = (i == 0) ? i : (i-1);
//      unsigned jp1 = (j == N-1) ? j : (j+1);
//      unsigned jm1 = (j == 0) ? j : (j-1);
//
//      double u2v = GET(u, i, j) * GET(u, i, j) * GET(v, i, j);
//      GETnb(du, i, j) = alpha*( GET(u, im1, j) + GET(u, ip1, j) + GET(u, i, jp1) + GET(u, i, jm1) - 4 * GET(u, i, j))
//                      + B + u2v - (A + 1)*GET(u, i, j) + brusselator_f(x, y, t);
//      GETnb(dv, i, j) = alpha*( GET(v, im1, j) + GET(v, ip1, j) + GET(v, i, jp1) + GET(v, i, jm1) - 4 * GET(v, i, j))
//                      + A * GET(u, i, j) - u2v;
//    }
//  }
//}

type state_type = [f64; 2 * N * N];

fn lorenz(x: &state_type, dxdt: &mut state_type, t: f64) {
    let p = [3.4, 1., 10.];
    let (tmp1, tmp2) = dxdt.split_at_mut(N * N);
    let mut dxdt1: [f64; N * N] = tmp1.try_into().unwrap();
    let mut dxdt2: [f64; N * N] = tmp2.try_into().unwrap();
    brusselator_2d_loop(&mut dxdt1, &mut dxdt2, &x[..], &x[N * N..], &p, t);
}

#[no_mangle]
pub extern "C" fn rust_dbrusselator_2d_loop(p: *const f64, x: *const state_type, adjoint: *mut state_type, t: f64) -> f64 {
    let x = unsafe { *x };
    let mut adjoint = unsafe { *adjoint };
    let p: [f64;3] = unsafe { *p.cast::<[f64;3]>().as_ref().unwrap() };
    let mut dp = [0.; 3];
    let mut dx1 = [0.; N * N];
    let mut dx2 = [0.; N * N];
    let (mut dadj1, mut dadj2) = adjoint.split_at_mut(N * N);

    let (tmp1, tmp2) = x.split_at(N * N);
    let x1: [f64; N * N] = tmp1.try_into().unwrap();
    let x2: [f64; N * N] = tmp2.try_into().unwrap();
    
    let mut null1 = [0.; 2 * N * N];
    let mut null2 = [0.; 2 * N * N];
    dbrusselator_2d_loop(&mut null1, &mut dadj1,
                         &mut null2, &mut dadj2,
                         &x1, &mut dx1, 
                         &x2, &mut dx2,
                         &p, &mut dp, t);
    dx1[0]
}


fn foobar(p: &[f64;3], x: state_type, mut adjoint: state_type, t: f64) -> f64 {
    let mut dp = [0.; 3];
    let mut dx1 = [0.; N * N];
    let mut dx2 = [0.; N * N];
    let (mut dadj1, mut dadj2) = adjoint.split_at_mut(N * N);
    let mut null1 = [0.; 2 * N * N];
    let mut null2 = [0.; 2 * N * N];
    let (tmp1, tmp2) = x.split_at(N * N);
    let x1: [f64; N * N] = tmp1.try_into().unwrap();
    let x2: [f64; N * N] = tmp2.try_into().unwrap();
    dbrusselator_2d_loop(&mut null1, &mut dadj1,
                         &mut null2, &mut dadj2,
                         &x1, &mut dx1, 
                         &x2, &mut dx2,
                         &p, &mut dp, t);
    dx1[0]
}

//double foobar(const double* p, const state_type x, const state_type adjoint, double t) {
//    double dp[3] = { 0. };
//
//    state_type dx = { 0. };
//
//    state_type dadjoint_inp = adjoint;
//
//    state_type dxdu;
//
//    __enzyme_autodiff<void>(brusselator_2d_loop,
//                            enzyme_dupnoneed, nullptr, dadjoint_inp.data(),
//                            enzyme_dupnoneed, nullptr, dadjoint_inp.data() + N * N,
//                            enzyme_dup, x.data(), dx.data(),
//                            enzyme_dup, x.data() + N * N, dx.data() + N * N,
//                            enzyme_dup, p, dp,
//                            enzyme_const, t);
//
//    return dx[0];
//}

fn main() {
    let p = [3.4, 1., 10.];
    let mut x = [0.; 2 * N * N];
    let mut adjoint = [0.; 2 * N * N];
    init_brusselator(&mut x, &mut adjoint);
    let t = 2.1;
    let mut res = 0.;
    let time = std::time::Instant::now();
    for _ in 0..10000 {
        res = foobar(&p, x, adjoint, t);
    }
    println!("Enzyme combined {} res={}", time.elapsed().as_secs_f64(), res);
}
