---
title: "Rust"
date: 2019-11-29T15:26:15Z
draft: false
weight: 40
---

Enzyme.jl can be installed in the usual way Julia packages are installed:

```sh
] add Enzyme
```

The Enzyme binary dependencies will be installed automatically via Julia's binary actifact system.

The Enzyme.jl API revolves around the function `autodiff`, see it's [documentation](https://enzyme.mit.edu/julia/api/#Enzyme.autodiff-Union{Tuple{A},%20Tuple{F},%20Tuple{F,%20Type{A},%20Vararg{Any}}}%20where%20{F,%20A%3C:Enzyme.Annotation}) for details and a usage example. 
Also see [Implementing pullbacks](https://enzyme.mit.edu/julia/pullbacks/) on how to use Enzyme.jl to implement back-propagation for functions with non-scalar results.
