use std::slice;
use std::f64::consts::PI;

fn bitreversal_perm<T>(data: &mut [T]) {
    let len = data.len() / 2;
    let mut j = 1;

    let mut i = 1;
    while i < 2*len {
        if j > i {
            //dbg!(&i, &j);
            unsafe {data.swap_unchecked(j-1, i-1);}
            unsafe {data.swap_unchecked(j, i);}
        }

        let mut m = len;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }

        j += m;
        i += 2;
    }
}

fn radix2(data: &mut [f64], i_sign: f64, n: usize) {
    if n == 1 {
        return;
    }

    let (a,b) = data.split_at_mut(n);
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

        let tempr = data[in_n] * wr - data[in_n + 1] * wi;
        let tempi = data[in_n] * wi + data[in_n + 1] * wr;

        data[in_n] = data[i] - tempr;
        data[in_n + 1] = data[i + 1] - tempi;
        data[i] += tempr;
        data[i + 1] += tempi;

        let wtemp_new = wr;
        wr += wr * wpr - wi * wpi;
        wi += wi * wpr + wtemp_new * wpi;

        i += 2;
    }
}

fn rescale(data: &mut [f64], scale: usize) {
    let scale = 1. / scale as f64;
    for elm in data {
        *elm *= scale;
    }
}

fn fft(data: &mut [f64]) {
    bitreversal_perm(data);
    radix2(data, 1.0, data.len() / 2);
}

fn ifft(data: &mut [f64]) {
    bitreversal_perm(data);
    radix2(data, -1.0, data.len() / 2);
    rescale(data, data.len() / 2);
}

#[autodiff(dfoobar, Reverse, Duplicated)]
pub fn foobar(data: &mut [f64]) {
    fft(data);
    ifft(data);
}

#[no_mangle]
pub extern "C" fn rust_dfoobar(n: usize, data: *mut f64, ddata: *mut f64) {
    
    let (data, ddata) = unsafe {
        (
            slice::from_raw_parts_mut(data, n * 2),
            slice::from_raw_parts_mut(ddata, n * 2)
        )
    };

    dfoobar(data, ddata);
}

#[no_mangle]
pub extern "C" fn rust_foobar(n: usize, data: *mut f64) {
    let data = unsafe { slice::from_raw_parts_mut(data, n * 2) };
    foobar(data);
}
