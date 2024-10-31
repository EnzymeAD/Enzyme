use std::autodiff::autodiff;
use std::f64::consts::PI;
use std::slice;

fn bitreversal_perm<T>(data: &mut [T]) {
    let len = data.len() / 2;
    let mut j = 1;

    let mut i = 1;
    while i < 2 * len {
        if j > i {
            //dbg!(&i, &j);
            //data.swap(j-1, i-1);
            //data.swap(j, i);
            unsafe {
                data.swap_unchecked(j - 1, i - 1);
            }
            unsafe {
                data.swap_unchecked(j, i);
            }
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

fn radix2(data: &mut [f64], i_sign: i32) {
    let n = data.len() / 2;
    if n == 1 {
        return;
    }

    let (a, b) = data.split_at_mut(n);
    radix2(a, i_sign);
    radix2(b, i_sign);

    let wtemp = i_sign as f64 * (PI / n as f64).sin();
    let wpi = -i_sign as f64 * (2.0 * PI / n as f64).sin();
    let wpr = -2.0 * wtemp * wtemp;
    let mut wr = 1.0;
    let mut wi = 0.0;

    for i in (0..n).step_by(2) {
        let tempr = b[i] * wr - b[i + 1] * wi;
        let tempi = b[i] * wi + b[i + 1] * wr;

        b[i] = a[i] - tempr;
        b[i + 1] = a[i + 1] - tempi;
        a[i] += tempr;
        a[i + 1] += tempi;

        let wtemp_new = wr;
        wr += wr * wpr - wi * wpi;
        wi += wi * wpr + wtemp_new * wpi;
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
    radix2(data, 1);
}

fn ifft(data: &mut [f64]) {
    bitreversal_perm(data);
    radix2(data, -1);
    rescale(data, data.len() / 2);
}

#[autodiff(dfoobar, Reverse, DuplicatedOnly)]
pub fn foobar(data: &mut [f64]) {
    fft(data);
    ifft(data);
}

#[no_mangle]
pub extern "C" fn rust_dfoobar(n: usize, data: *mut f64, ddata: *mut f64) {
    let (data, ddata) = unsafe {
        (
            slice::from_raw_parts_mut(data, n * 2),
            slice::from_raw_parts_mut(ddata, n * 2),
        )
    };

    unsafe { dfoobar(data, ddata) };
}

#[no_mangle]
pub extern "C" fn rust_foobar(n: usize, data: *mut f64) {
    let data = unsafe { slice::from_raw_parts_mut(data, n * 2) };
    foobar(data);
}
