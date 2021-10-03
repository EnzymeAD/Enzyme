---
date: 2017-10-19T15:26:15Z
lastmod: 2019-10-26T15:26:15Z
publishdate: 2018-11-23T15:26:15Z
---

# Enzyme Overview

The Enzyme project is a tool for performing reverse-mode automatic differentiation (AD) of statically-analyzable LLVM IR. This allows developers to use Enzyme to automatically create gradients of their source code without much additional work.

```c
double foo(double);

double grad_foo(double x) {
    return __enzyme_autodiff(foo, x);
}
```

By differentiating code after optimization, Enzyme is able to create substantially faster derivatives than existing tools that differentiate programs before optimization.

<div style="padding:2em">
<img src="/all_top.png" width="500" align=center>
</div>

## Components

Enzyme is composed of four pieces:

*   An optional preprocessing phase which performs minor transformations that tend to be helpful for AD.
*   A new interprocedural type analysis that deduces the underlying types of memory locations
*   An activity analaysis that determines what instructions or values can impact the derivative computation (common in existing AD systems).
*   An optimization pass which creates any required derivative functions, replacing calls to `__enzyme_autodiff` with the generated functions.

## More resources

For more information on Enzyme, please see:

*   The Enzyme [getting started guide](/getting_started/)
*   The Enzyme [mailing list](https://groups.google.com/d/forum/enzyme-dev) for any questions.
*   Previous [talks](/talks/).
*   You can try out Enzyme on our [Compiler Explorer instance](/explorer).

## Citing Enzyme

To cite Enzyme, please cite the following:
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

The original Enzyme is also avaiable as a preprint on [arXiv](https://arxiv.org/pdf/2010.01709.pdf).
