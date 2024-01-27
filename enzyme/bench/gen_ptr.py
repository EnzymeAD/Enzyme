import os
import argparse
import numpy as np

def create_all_f(out_path, n_funcs, n_lines):
    f_out = open(out_path, 'w+')
    f_out.write('#include <vector>\n')
    f_out.write('#include <cstdlib>\n')
    f_out.write('using std::vector;\n\n')
    f_out.close()
    for i in range(n_funcs):
        create_f(out_path, i, n_lines, max(10,i*n_lines))
    f_out = open(out_path, 'a+')
    f_out.write('void f(float* vec) {\n')
    for i in range(n_funcs):
        f_out.write('    f_{}(vec);\n'.format(i))
    f_out.write('    return;\n')
    f_out.write('}\n\n')

def create_f(out_path, block, n_lines, offset):
    f_out = open(out_path, 'a+')
    ops = ['+', '-', '*'] #, '/' too likely to cause issues
    
    f_out.write('void f_{}(float* vec) {{\n'.format(block))
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

    f_out.write('void __enzyme_autodiff(void*, float* vec, float* dvec);\n')
    f_out.write('int main() {\n')
    f_out.write('    std::vector<float> vec, dvec;\n')
    f_out.write('    vec.resize({});\n'.format(n_funcs*n_lines))
    f_out.write('    dvec.resize({});\n'.format(n_funcs*n_lines))
    
    # initialize first 10 entries randomly
    # C++ again -.-
    for i in range(10):
        f_out.write('    vec[{}] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);\n'.format(i))
    
    f_out.write('    __enzyme_autodiff((void*)f, vec.data(), dvec.data());\n')

    f_out.write('    return 0;\n')
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

    file_mono = 'stress_{}_{}_ptr.cpp'.format(n_funcs, n_lines)
    create_all_f(file_mono, n_funcs, n_lines)
    create_main(file_mono, n_funcs, n_lines)
    

if __name__ == "__main__":

    main()
    
