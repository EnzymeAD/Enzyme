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

## Citing Enzyme

To cite Enzyme, please use [this publication](/enzymepreprint.pdf).
```
@incollection{enzymeNeurips,
title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
author = {Moses, William S. and Churavy, Valentin},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2020},
note = {To appear in},
}
```
