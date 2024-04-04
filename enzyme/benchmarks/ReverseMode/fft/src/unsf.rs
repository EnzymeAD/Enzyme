use std::f64::consts::PI;

//static void scramble(double* data, unsigned N) {
//  int j=1;
//  for (int i=1; i<2*N; i+=2) {
//    if (j>i) {
//      swap(&data[j-1], &data[i-1]);
//      swap(&data[j], &data[i]);
//    }
//    int m = N;
//    while (m>=2 && j>m) {
//      j -= m;
//      m >>= 1;
//    }
//    j += m;
//  }
//}
unsafe fn bitreversal_perm(data: *mut f64, n: usize) {
    //let len = data.len() / 2;
    let mut j = 1;

    for i in (1..2*n).step_by(2) {
    //let mut i = 1;
    //while i < 2*len {
        if j > i {
            std::ptr::swap(data.add(j-1), data.add(i-1));
            std::ptr::swap(data.add(j), data.add(i));
            //data.swap(j-1, i-1);
            //data.swap(j, i);
        }

        let mut m = n;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }

        j += m;
        //i += 2;
    }
}

unsafe fn radix2(data: *mut f64, i_sign: f64, n: usize) {
    if n == 1 {
        return;
    }

    let b = data.add(n);
    let a = data;
    //let (a,b) = data.split_at_mut(n);
    radix2(a, i_sign, n/2);
    radix2(b, i_sign, n/2);

    let wtemp = i_sign * (PI / n as f64).sin();
    let wpi = -i_sign * (2.0 * PI / n as f64).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let mut wr = 1.0;
    let mut wi = 0.0;

    let mut i = 0;
    while i < n {
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

        i += 2;
    }
}

//static void rescale(double* data, unsigned N) {
//  double scale = ((double)1)/N;
//  for (unsigned i=0; i<2*N; i++) {
//    data[i] *= scale;
//  }
//}

unsafe fn rescale(data: *mut f64, n: usize) {
    let scale = 1. / n as f64;
    for i in 0..2*n {
        *data.add(i) = *data.add(i) * scale;
    }
}

unsafe fn fft(data: *mut f64, n: usize) {
    bitreversal_perm(data, n);
    radix2(data, 1.0, n);
}

unsafe fn ifft(data: *mut f64, n: usize) {
    bitreversal_perm(data, n);
    radix2(data, -1.0, n);
    rescale(data, n);
}

#[autodiff(unsafe_dfoobar, Reverse, Const, Duplicated)]
pub unsafe fn unsafe_foobar(n: usize, data: *mut f64) {
    fft(data, n / 2);
    ifft(data, n / 2);
}

#[no_mangle]
pub extern "C" fn rust_unsafe_dfoobar(n: usize, data: *mut f64, ddata: *mut f64) {
    unsafe {unsafe_dfoobar(n, data, ddata); }
}

#[no_mangle]
pub extern "C" fn rust_unsafe_foobar(n: usize, data: *mut f64) {
    unsafe {unsafe_foobar(n, data); }
}
