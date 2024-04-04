use fft::dfoobar;

fn main() {
    let mut data = vec![1.0; 32];
    for i in 0..16 {
        data[i] = 2.0;
    }
    let mut data_d = vec![1.0; data.len()];

    dfoobar(&mut data, &mut data_d);

    dbg!(&data_d);
    dbg!(&data);
}
