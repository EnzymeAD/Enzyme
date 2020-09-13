---
title: "Open Projects"
date: 2019-11-29T15:26:15Z
draft: false
weight: 25
---

Below is a list of projects that can be suitable for [Google Summer of Code
(GSOC)](https://summerofcode.withgoogle.com/) or just for someone to get started
with contributing to MLIR. See also [the "beginner" issues](https://bugs.llvm.org/buglist.cgi?keywords=beginner%2C%20&keywords_type=allwords&list_id=176893&product=MLIR&query_format=advanced&resolution=---)
on the bugtracker.
If you're interested in one of these projects, feel free to discuss it on
the MLIR section of the [LLVM forums](https://llvm.discourse.group/c/mlir/31)
or on the MLIR channel of the [LLVM discord](https://discord.gg/xS7Z362)
server. The mentors are indicative and suggestion of first point of contact for
starting on these projects.

* Implement C bindings for the core IR: this will allow to manipulate IR from other languages.
* llvm-canon kind of tools for MLIR (mentor: Mehdi Amini, Jacques Pienaar)
* IR query tool to make exploring the IR easier (e.g., all operations dominated
  by X, find possible path between two ops, etc.) (mentor: Jacques Pienaar)
* Automatic formatter for TableGen (similar to clang-format for C/C++)
* LLVM IR declaratively defined. (mentor: Alex Zinenko)
* MLIR Binary serialization / bitcode format (Mehdi Amini)
* SPIR-V module combiner (mentor: Lei Zhang)
  * Basic: merging modules and remove identical functions
  * Advanced: comparing logic and use features like spec constant to reduce
  similar but not identical functions
* GLSL to SPIR-V dialect frontend (mentor: Lei Zhang)
  * Requires: building up graphics side of the SPIR-V dialect
  * Purpose: give MLIR more frontends :) improve graphics tooling
  * Potential real-world usage: providing a migration solution from WebGL
  (shaders represented as GLSL) to WebGPU (shaders represented as SPIR-V-like language, [WGSL](https://gpuweb.github.io/gpuweb/wgsl.html))
* TableGen "front-end dialect" (mentor: Jacques Pienaar)
* Making MLIR interact with existing polyhedral tools: isl, pluto (mentor: Alex Zinenko)
* MLIR visualization (mentor: Jacques Pienaar)

  MLIR allows for representing multiple levels of abstraction all together in the same IR/function. Visualizing MLIR modules therefore requires going beyond visualizing a graph of nodes all at the same level (which is not trivial in and of itself!), nor is it specific to Machine Learning. Beyond visualizing a MLIR module, there is also visualizing MLIR itself that is of interest. In particular, visualizing the rewrite rules, visualizing the matching process (including the failure to match, sort of like https://www.debuggex.com/ but for declarative rewrites), considering effects of rewrites over time, etc.

  The visualizations should all be built with open source components but whether standalone (e.g., combining with, say, GraphViz to generate offline images) or dynamic tools (e.g., displayed in browser) is open for discussion. It should be usable completely offline in either case.
	
	We will be working with interested students to refine the exact project based on interests given the wide scope of potential approaches. And open to proposals within this general area.

* Rewrite patterns expressed in MLIR (mentor: Jacques Pienaar)
* Generic value range analysis for MLIR (mentor: River Riddle)

### Projects started/starting soon:

This is section for projects that have not yet started but there are
individuals/groups intending to start work on in near future.

* Rework the MLIR python bindings, add a C APIs for core concepts (mentor: 
  Nicolas Vasilache, Alex Zinenko)
* [bugpoint/llvm-reduce](https://llvm.org/docs/BugpointRedesign.html) kind
  of tools for MLIR (mentor: Mehdi Amini, Jacques Pienaar)
* MLIR visualization, there are some projects in flight but we unfortunately
  don't know the project plans of those teams. But if you intend to work on
	something in this area it would be good to discuss on the forum early
	in case there are collaboration opportunity.
  
