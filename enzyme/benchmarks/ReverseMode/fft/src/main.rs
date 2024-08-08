use core::mem;
use fft::safe;//::dfoobar;
use fft::unsf;//::dfoobar;

fn main() {
    let len = 16;
    let mut data = vec![1.0; 2*len];
    for i in 0..len {
        data[i] = 2.0;
    }
    let mut data_d = vec![1.0; 2*len];

    //unsafe {safe::rust_dfoobar(len, data.as_mut_ptr(), data_d.as_mut_ptr());}
    //unsafe {safe::rust_foobar(len, data.as_mut_ptr());}
    unsafe {unsf::unsafe_dfoobar(len, data.as_mut_ptr(), data_d.as_mut_ptr());}
    unsafe {unsf::unsafe_foobar(len, data.as_mut_ptr());}

    dbg!(&data_d);
    dbg!(&data);
    //mem::forget(data);
    //mem::forget(data_d);
}
