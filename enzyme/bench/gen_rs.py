import os
import argparse
import numpy as np

def create_all_f(out_path, n_funcs, n_lines):
    f_out = open(out_path, 'w+')
    f_out.write('//#![feature(rustc_attrs)]\n')
    f_out.write('//#![feature(autodiff)]\n')
    f_out.write('use std::hint::black_box;\n\n')
    f_out.close()
    for i in range(n_funcs):
        create_f(out_path, i, n_lines, max(10,i*n_lines), n_funcs)
    f_out = open(out_path, 'a+')
    f_out.write('#[autodiff(df, Reverse, Duplicated)]\n')
    f_out.write('fn f(vec: &mut [f32]) {\n')
    for i in range(n_funcs):
        f_out.write('    f_{}(vec);\n'.format(i))
    f_out.write('}\n\n')

def create_f(out_path, block, n_lines, offset, n_funcs):
    f_out = open(out_path, 'a+')
    ops = ['+', '-', '*'] #, '/' too likely to cause issues
    
    f_out.write('fn f_{}(vec: &mut [f32]) {{\n'.format(block))
    f_out.write('    assert!(vec.len() > {});\n'.format(n_lines * n_funcs))
    for i in range(n_lines):
        str_ = '    vec[{}] = '.format(i+offset)
        for j in range(2):
            str_ += 'vec[{}]'.format(np.random.randint(0, i+offset))
            str_ += ' {} '.format(ops[np.random.randint(0, len(ops))])
        str_ += 'vec[{}]'.format(np.random.randint(0, i+offset))
        f_out.write('{};\n'.format(str_))
    f_out.write('}\n\n')
    f_out.close()

def create_main(file_mono, n_funcs, n_lines):
    f_out = open(file_mono, 'a+')

    f_out.write('fn main() {\n')
    f_out.write('    let mut vec = vec![0.0; {}];\n'.format(n_funcs*n_lines))
    f_out.write('    let mut dvec = vec![0.0; {}];\n'.format(n_funcs*n_lines))
    f_out.write('    black_box(vec[0..10].fill(3.14));\n')
    f_out.write('    f(&mut vec);\n')
    f_out.write('    // df(&mut vec, &mut dvec);\n')
    f_out.write('    black_box(vec);\n')
    f_out.write('    black_box(dvec);\n')
    f_out.write('}\n')
    f_out.close()


# ----------------------------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(description='Stresstesting Enzyme')
    parser.add_argument('-num', type=int, default='5', help='number of subfunctions')
    parser.add_argument('-len', type=int, default='500', help='lines per subfunction')
    args = parser.parse_args()
    
    n_funcs = args.num
    n_lines = args.len
    assert(n_funcs * n_lines > 10)

    file_mono = 'stress_{}_{}.rs'.format(n_funcs, n_lines)
    create_all_f(file_mono, n_funcs, n_lines)
    create_main(file_mono, n_funcs, n_lines)
    

if __name__ == "__main__":

    main()
    
