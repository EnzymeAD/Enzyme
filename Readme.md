# <img src="https://enzyme.mit.edu/logo.svg" width="75" align=left> The Enzyme High-Performance Automatic Differentiator of LLVM


Enzyme is a plugin that performs automatic differentiation (AD) of statically analyzable LLVM.

Enzyme can be used by calling `__enzyme_autodiff` on a function to be differentiated as shown below. 
Running the Enzyme transformation pass then replaces the call to `__enzyme_autodiff` with the gradient of its first argument.
```c
double foo(double);

double grad_foo(double x) {
    return __enzyme_autodiff(foo, x);
}
```

Enzyme is highly-efficient and its ability to perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

<div style="padding:2em">
<img src="https://enzyme.mit.edu/all_top.png" width="500" align=center>
</div>

Detailed information on installing and using Enzyme can be found on our website: [https://enzyme.mit.edu](https://enzyme.mit.edu).

A short example of how to install Enzyme is below:
```
cd /path/to/Enzyme/enzyme
mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DLLVM_EXTERNAL_LIT=/path/to/lit/lit.py
ninja
```

Or, install Enzyme using [Homebrew](https://brew.sh):
```
brew install enzyme
```

To get involved or if you have questions, please join our [mailing list](https://groups.google.com/d/forum/enzyme-dev).

If using this code in an academic setting, please cite the following:
```
@inproceedings{NEURIPS2020_9332c513,
 author = {Moses, William and Churavy, Valentin},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {12472--12485},
 publisher = {Curran Associates, Inc.},
 title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
 url = {https://proceedings.neurips.cc/paper/2020/file/9332c513ef44b682e9347822c2e457ac-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

Julia bindings for Enzyme are available [here](https://github.com/wsmoses/Enzyme.jl)

