use std::f64::consts::PI;
use std::autodiff::autodiff;

unsafe fn bitreversal_perm(data: *mut f64, len: usize) {
    let mut j = 1;

    for i in (1..2*len).step_by(2) {
        if j > i {
            std::ptr::swap(data.add(j-1), data.add(i-1));
            std::ptr::swap(data.add(j), data.add(i));
        }

        let mut m = len;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }

        j += m;
    }
}

unsafe fn radix2(data: *mut f64, i_sign: i32, n: usize) {
    if n == 1 { return; }
    radix2(data, i_sign, n/2);
    radix2(data.add(n), i_sign, n/2);

    let wtemp = i_sign as f64 * (PI / n as f64).sin();
    let wpi = -i_sign as f64 * (2.0 * PI / n as f64).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let mut wr = 1.0;
    let mut wi = 0.0;

    for i in (0..n).step_by(2) {
        let in_n = i + n;

        let tempr = *data.add(in_n) * wr - *data.add(in_n + 1) * wi;
        let tempi = *data.add(in_n) * wi + *data.add(in_n + 1) * wr;

        *data.add(in_n) = *data.add(i) - tempr;
        *data.add(in_n + 1) = *data.add(i + 1) - tempi;
        *data.add(i) += tempr;
        *data.add(i + 1) += tempi;

        let wtemp_new = wr;
        wr += wr * wpr - wi * wpi;
        wi += wi * wpr + wtemp_new * wpi;
    }
}

unsafe fn rescale(data: *mut f64, n: usize) {
    let scale = 1. / n as f64;
    for i in 0..2*n {
        *data.add(i) = *data.add(i) * scale;
    }
}

unsafe fn fft(data: *mut f64, n: usize) {
    bitreversal_perm(data, n);
    radix2(data, 1, n);
}

unsafe fn ifft(data: *mut f64, n: usize) {
    bitreversal_perm(data, n);
    radix2(data, -1, n);
    rescale(data, n);
}

#[autodiff(unsafe_dfoobar, Reverse, Const, DuplicatedOnly)]
pub unsafe fn unsafe_foobar(n: usize, data: *mut f64) {
    fft(data, n );
    ifft(data, n );
}

#[no_mangle]
pub extern "C" fn rust_unsafe_dfoobar(n: usize, data: *mut f64, ddata: *mut f64) {
    unsafe {unsafe_dfoobar(n, data, ddata); }
}

#[no_mangle]
pub extern "C" fn rust_unsafe_foobar(n: usize, data: *mut f64) {
    unsafe {unsafe_foobar(n, data); }
}
