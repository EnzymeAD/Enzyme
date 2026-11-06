# Variable Names Plan

> **Note:** This plan is *provisional*. It is not agreed upon. It is written with the intention of capturing the desires and concerns of the LLVM community, and forming them into a plan that can be agreed upon.
>
> The original author is somewhat naïve in the ways of LLVM so there will inevitably be some details that are flawed. You can help - you can edit this page (preferably with a Phabricator review for larger changes) or reply to the [Request For Comments thread](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130083.html).

---

## Table of Contents

- [Too Long; Didn't Read](#too-long-didnt-read)
- [Introduction](#introduction)
- [Variable Names Coding Standard Options](#variable-names-coding-standard-options)
  - [Differentiating variable kinds](#differentiating-variable-kinds)
- [Reducing the number of acronyms](#reducing-the-number-of-acronyms)
- [Transition Options](#transition-options)
  - [Keep the current coding standard](#keep-the-current-coding-standard)
  - [Laissez faire](#laissez-faire)
  - [Big bang](#big-bang)
    - [Keeping git blame usable](#keeping-git-blame-usable)
    - [Minimising cost of downstream merges](#minimising-cost-of-downstream-merges)
- [Provisional Plan](#provisional-plan)
- [References](#references)

---

## Too Long; Didn't Read

Improve the readability of LLVM code.

---

## Introduction

The current [variable naming rule](../CodingStandards.html#name-types-functions-variables-and-enumerators-properly) states:

> Variable names should be nouns (as they represent state). The name should be camel case, and start with an upper case letter (e.g. `Leader` or `Boats`).

This rule is the same as that for type names. This is a problem because the type name cannot be reused for a variable name [[*](#footnote-shadowing)]. LLVM developers tend to work around this by either prepending `The` to the type name:

```cpp
Triple TheTriple;
```

... or more commonly use an acronym, despite the coding standard stating "Avoid abbreviations unless they are well known":

```cpp
Triple T;
```

The proliferation of acronyms leads to hard-to-read code such as [this](https://github.com/llvm/llvm-project/blob/7aecb64cf6b5093eb9d59e846061ad349fedfb5a/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L7050):

```cpp
InnerLoopVectorizer LB(L, PSE, LI, DT, TLI, TTI, AC, ORE, VF.Width, IC, &LVL, &CM);
```

Many other coding guidelines ([LLDB](#ref-lldb), [Google](#ref-google), [WebKit](#ref-webkit), [Qt](#ref-qt), [Rust](#ref-rust), [Swift](#ref-swift), [Python](#ref-python)) require that variable names begin with a lower case letter in contrast to class names which begin with a capital letter. This convention means that the most readable variable name also requires the least thought:

```cpp
Triple triple;
```

There is some agreement that the current rule is broken ([LattnerAgree](#ref-lattneragree), [ArsenaultAgree](#ref-arsenaultagree), [RobinsonAgree](#ref-robinsonagree)) and that acronyms are an obstacle to reading new code ([MalyutinDistinguish](#ref-malyutindistinguish), [CarruthAcronym](#ref-carruthacronym), [PicusAcronym](#ref-picusacronym)). There are some opposing views ([ParzyszekAcronym2](#ref-parzyszekacronym2), [RicciAcronyms](#ref-ricciacronyms)).

This work-in-progress proposal is to change the coding standard for variable names to require that they start with a lower case letter.

<a id="footnote-shadowing"></a>[*] In [some cases](https://github.com/llvm/llvm-project/blob/8b72080d4d7b13072f371712eed333f987b7a18e/llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp#L2727) the type name *is* reused as a variable name, but this shadows the type name and confuses many debuggers ([DenisovCamelBack](#ref-denisovcamelback)).

---

## Variable Names Coding Standard Options

There are two main options for variable names that begin with a lower case letter: `camelBack` and `lower_case`. (These are also known by other names but here we use the terminology from `clang-tidy`).

- `camelBack` is consistent with [WebKit](#ref-webkit), [Qt](#ref-qt) and [Swift](#ref-swift).
- `lower_case` is consistent with [LLDB](#ref-lldb), [Google](#ref-google), [Rust](#ref-rust) and [Python](#ref-python).

`camelBack` is already used for function names, which may be considered an advantage ([LattnerFunction](#ref-lattnerfunction)) or a disadvantage ([CarruthFunction](#ref-carruthfunction)).

- Approval for `camelBack` was expressed by [DenisovCamelBack](#ref-denisovcamelback), [LattnerFunction](#ref-lattnerfunction), and [IvanovicDistinguish](#ref-ivanovicdistinguish).
- Opposition to `camelBack` was expressed by [CarruthCamelBack](#ref-carruthcamelback) and [TurnerCamelBack](#ref-turnercamelback).
- Approval for `lower_case` was expressed by [CarruthLower](#ref-carruthlower), [CarruthCamelBack](#ref-carruthcamelback), and [TurnerLLDB](#ref-turnerlldb).
- Opposition to `lower_case` was expressed by [LattnerLower](#ref-lattnerlower).

### Differentiating variable kinds

An additional requested change is to distinguish between different kinds of variables ([RobinsonDistinguish](#ref-robinsondistinguish), [RobinsonDistinguish2](#ref-robinsondistinguish2), [JonesDistinguish](#ref-jonesdistinguish), [IvanovicDistinguish](#ref-ivanovicdistinguish), [CarruthDistinguish](#ref-carruthdistinguish), [MalyutinDistinguish](#ref-malyutindistinguish)).

Others oppose this idea ([HähnleDistinguish](#ref-haehnledistinguish), [GreeneDistinguish](#ref-greenedistinguish), [HendersonPrefix](#ref-hendersonprefix)).

- A possibility is for member variables to be prefixed with `m_` and for global variables to be prefixed with `g_` to distinguish them from local variables. This is consistent with [LLDB](#ref-lldb). The `m_` prefix is consistent with [WebKit](#ref-webkit).
- A variation is for member variables to be prefixed with `m` ([IvanovicDistinguish](#ref-ivanovicdistinguish), [BeylsDistinguish](#ref-beylsdistinguish)). This is consistent with [Mozilla](#ref-mozilla).
- Another option is for member variables to be suffixed with `_` which is consistent with [Google](#ref-google) and similar to [Python](#ref-python). Opposed by [ParzyszekDistinguish](#ref-parzyszekdistinguish).

---

## Reducing the number of acronyms

While switching coding standard will make it easier to use non-acronym names for new code, it doesn't improve the existing large body of code that uses acronyms extensively to the detriment of its readability. Further, it is natural and generally encouraged that new code be written in the style of the surrounding code. Therefore it is likely that much newly written code will also use acronyms despite what the coding standard says, much as it is today.

As well as changing the case of variable names, they could also be expanded to their non-acronym form e.g. `Triple T` → `Triple triple`.

There is support for expanding many acronyms ([CarruthAcronym](#ref-carruthacronym), [PicusAcronym](#ref-picusacronym)) but there is a preference that expanding acronyms be deferred ([ParzyszekAcronym](#ref-parzyszekacronym), [CarruthAcronym](#ref-carruthacronym)).

The consensus within the community seems to be that at least some acronyms are valuable ([ParzyszekAcronym](#ref-parzyszekacronym), [LattnerAcronym](#ref-lattneracronym)). The most commonly cited acronym is `TLI` however that is used to refer to both `TargetLowering` and `TargetLibraryInfo` ([GreeneDistinguish](#ref-greenedistinguish)).

The following is a list of acronyms considered sufficiently useful that the benefit of using them outweighs the cost of learning them. Acronyms that are either not on the list or are used to refer to a different type should be expanded.

| Class name | Variable name |
| :--- | :--- |
| `DeterministicFiniteAutomaton` | `dfa` |
| `DominatorTree` | `dt` |
| `LoopInfo` | `li` |
| `MachineFunction` | `mf` |
| `MachineInstr` | `mi` |
| `MachineRegisterInfo` | `mri` |
| `ScalarEvolution` | `se` |
| `TargetInstrInfo` | `tii` |
| `TargetLibraryInfo` | `tli` |
| `TargetRegisterInfo` | `tri` |

In some cases renaming acronyms to the full type name will result in overly verbose code. Unlike most classes, a variable's scope is limited and therefore some of its purpose can implied from that scope, meaning that fewer words are necessary to give it a clear name. For example, in an optimization pass the reader can assume that a variable's purpose relates to optimization and therefore an `OptimizationRemarkEmitter` variable could be given the name `remarkEmitter` or even `remarker`.

| Full Class Name | Concise Proposed Name |
| :--- | :--- |
| `OptimizationRemarkEmitter` | `remarkEmitter` / `remarker` |
| `PreservedAnalyses` | `analyses` |
| `PreservedAnalysesChecker` | `checker` |
| `TargetLowering` | `lowering` |
| `TargetMachine` | `machine` |

---

## Transition Options

There are three main options for transitioning:

1. [Keep the current coding standard](#keep-the-current-coding-standard)
2. [Laissez faire](#laissez-faire)
3. [Big bang](#big-bang)

### Keep the current coding standard

Proponents of keeping the current coding standard (i.e. not transitioning at all) question whether the cost of transition outweighs the benefit ([EmersonConcern](#ref-emersonconcern), [ReamesConcern](#ref-reamesconcern), [BradburyConcern](#ref-bradburyconcern)).

The costs are that `git blame` will become less usable; and that merging the changes will be costly for downstream maintainers. See [Big bang](#big-bang) for potential mitigations.

### Laissez faire

The coding standard could allow both `CamelCase` and `camelBack` styles for variable names ([LattnerTransition](#ref-lattnertransition)).

A code review to implement this is at [D57896](https://reviews.llvm.org/D57896).

#### Advantages
- Very easy to implement initially.

#### Disadvantages
- Leads to inconsistency ([BradburyConcern](#ref-bradburyconcern), [AminiInconsistent](#ref-aminiinconsistent)).
- Inconsistency means it will be hard to know at a guess what name a variable will have ([DasInconsistent](#ref-dasinconsistent), [CarruthInconsistent](#ref-carruthinconsistent)).
- Some large-scale renaming may happen anyway, leading to its disadvantages without any mitigations.

### Big bang

With this approach, variables will be renamed by an automated script in a series of large commits.

The principle advantage of this approach is that it minimises the cost of inconsistency ([BradburyTransition](#ref-bradburytransition), [RobinsonTransition](#ref-robinsontransition)).

It goes against a policy of avoiding large-scale reformatting of existing code ([GreeneDistinguish](#ref-greenedistinguish)).

It has been suggested that LLD would be a good starter project for the renaming ([Ueyama](#ref-ueyama)).

#### Keeping git blame usable

`git blame` (or `git annotate`) permits quickly identifying the commit that changed a given line in a file. After renaming variables, many lines will show as being changed by that one commit, requiring a further invocation of `git blame` to identify prior, more interesting commits ([GreeneGitBlame](#ref-greenegitblame), [RicciAcronyms](#ref-ricciacronyms)).

**Mitigation**: [`git-hyper-blame`](https://commondatastorage.googleapis.com/chrome-infra-docs/flat/depot_tools/docs/html/git-hyper-blame.html) can ignore or "look through" a given set of commits. A `.git-blame-ignore-revs` file identifying the variable renaming commits could be added to the LLVM git repository root directory. It is being [investigated](https://public-inbox.org/git/20190324235020.49706-1-michael@platin.gs/) whether similar functionality could be added to `git blame` itself.

#### Minimising cost of downstream merges

There are many forks of LLVM with downstream changes. Merging a large-scale renaming change could be difficult for the fork maintainers.

**Mitigation**: A large-scale renaming would be automated. A fork maintainer can merge from the commit immediately before the renaming, then apply the renaming script to their own branch. They can then merge again from the renaming commit, resolving all conflicts by choosing their own version. This could be tested on the [SVE](#ref-sve) fork.

---

## Provisional Plan

This is a provisional plan for the [Big bang](#big-bang) approach. It has not been agreed.

1. Investigate improving `git blame`. The extent to which it can be made to "look through" commits may impact how big a change can be made.
2. Write a script to expand acronyms.
3. Experiment and perform dry runs of the various refactoring options. Results can be published in forks of the LLVM Git repository.
4. Consider the evidence and agree on the new policy.
5. Agree & announce a date for the renaming of the starter project (LLD).
6. Update the [policy page](../CodingStandards.html). This will explain the old and new rules and which projects each applies to.
7. Refactor the starter project in two commits:
   1. Add or change the project's `.clang-tidy` to reflect the agreed rules. (This is in a separate commit to enable the merging process described in [Minimising cost of downstream merges](#minimising-cost-of-downstream-merges)). Also update the project list on the policy page.
   2. Apply `clang-tidy` to the project's files, with only the `readability-identifier-naming` rules enabled. `clang-tidy` will also reformat the affected lines according to the rules in `.clang-format`. It is anticipated that this will be a good dog-fooding opportunity for `clang-tidy`, and bugs should be fixed in the process, likely including:
      - [readability-identifier-naming incorrectly fixes lambda capture](https://bugs.llvm.org/show_bug.cgi?id=41119)
      - [readability-identifier-naming incorrectly fixes variables which become keywords](https://bugs.llvm.org/show_bug.cgi?id=41120)
      - [readability-identifier-naming misses fixing member variables in destructor](https://bugs.llvm.org/show_bug.cgi?id=41122)
8. Gather feedback and refine the process as appropriate.
9. Apply the process to the following projects, with a suitable delay between each (at least 4 weeks after the first change, at least 2 weeks subsequently) to allow gathering further feedback. This list should exclude projects that must adhere to an externally defined standard e.g. `libcxx`. The list is roughly in chronological order of renaming. Some items may not make sense to rename individually - it is expected that this list will change following experimentation:

   1. TableGen
   2. llvm/tools
   3. clang-tools-extra
   4. clang
   5. ARM backend
   6. AArch64 backend
   7. AMDGPU backend
   8. ARC backend
   9. AVR backend
   10. BPF backend
   11. Hexagon backend
   12. Lanai backend
   13. MIPS backend
   14. NVPTX backend
   15. PowerPC backend
   16. RISC-V backend
   17. Sparc backend
   18. SystemZ backend
   19. WebAssembly backend
   20. X86 backend
   21. XCore backend
   22. libLTO
   23. Debug Information
   24. Remainder of llvm
   25. compiler-rt
   26. libunwind
   27. openmp
   28. parallel-libs
   29. polly
   30. lldb

10. Remove the old variable name rule from the policy page.

11. Repeat many of the steps in the sequence, using a script to expand acronyms.

---

## References

<a id="ref-lldb"></a>
- **[LLDB]** [LLDB Coding Conventions](https://llvm.org/svn/llvm-project/lldb/branches/release_39/www/lldb-coding-conventions.html)

<a id="ref-google"></a>
- **[Google]** [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#Variable_Names)

<a id="ref-webkit"></a>
- **[WebKit]** [WebKit Code Style Guidelines](https://webkit.org/code-style-guidelines/#names)

<a id="ref-qt"></a>
- **[Qt]** [Qt Coding Style](https://wiki.qt.io/Qt_Coding_Style#Declaring_variables)

<a id="ref-rust"></a>
- **[Rust]** [Rust naming conventions](https://doc.rust-lang.org/1.0.0/style/style/naming/README.html)

<a id="ref-swift"></a>
- **[Swift]** [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/#general-conventions)

<a id="ref-python"></a>
- **[Python]** [Style Guide for Python Code (PEP 8)](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names)

<a id="ref-mozilla"></a>
- **[Mozilla]** [Mozilla Coding style: Prefixes](https://developer.mozilla.org/en-US/docs/Mozilla/Developer_guide/Coding_Style#Prefixes)

<a id="ref-sve"></a>
- **[SVE]** [LLVM with support for SVE](https://github.com/ARM-software/LLVM-SVE)

<a id="ref-aminiinconsistent"></a>
- **[AminiInconsistent]** Mehdi Amini, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130329.html)

<a id="ref-arsenaultagree"></a>
- **[ArsenaultAgree]** Matt Arsenault, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/129934.html)

<a id="ref-beylsdistinguish"></a>
- **[BeylsDistinguish]** Kristof Beyls, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130292.html)

<a id="ref-bradburyconcern"></a>
- **[BradburyConcern]** Alex Bradbury, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130266.html)

<a id="ref-bradburytransition"></a>
- **[BradburyTransition]** Alex Bradbury, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130388.html)

<a id="ref-carruthacronym"></a>
- **[CarruthAcronym]** Chandler Carruth, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130313.html)

<a id="ref-carruthcamelback"></a>
- **[CarruthCamelBack]** Chandler Carruth, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130198.html)

<a id="ref-carruthdistinguish"></a>
- **[CarruthDistinguish]** Chandler Carruth, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130219.html)

<a id="ref-carruthfunction"></a>
- **[CarruthFunction]** Chandler Carruth, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130195.html)

<a id="ref-carruthinconsistent"></a>
- **[CarruthInconsistent]** Chandler Carruth, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130310.html)

<a id="ref-carruthlower"></a>
- **[CarruthLower]** Chandler Carruth, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130314.html)

<a id="ref-dasinconsistent"></a>
- **[DasInconsistent]** Sanjoy Das, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130304.html)

<a id="ref-denisovcamelback"></a>
- **[DenisovCamelBack]** Alex Denisov, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130179.html)

<a id="ref-emersonconcern"></a>
- **[EmersonConcern]** Amara Emerson, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/129894.html)

<a id="ref-greenedistinguish"></a>
- **[GreeneDistinguish]** David Greene, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130311.html)

<a id="ref-greenegitblame"></a>
- **[GreeneGitBlame]** David Greene, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130379.html)

<a id="ref-haehnledistinguish"></a>
- **[HähnleDistinguish]** Nicolai Hähnle, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/129923.html)

<a id="ref-hendersonprefix"></a>
- **[HendersonPrefix]** James Henderson, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130283.html)

<a id="ref-ivanovicdistinguish"></a>
- **[IvanovicDistinguish]** Nemanja Ivanovic, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130249.html)

<a id="ref-jonesdistinguish"></a>
- **[JonesDistinguish]** JD Jones, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/129926.html)

<a id="ref-lattneracronym"></a>
- **[LattnerAcronym]** Chris Lattner, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130193.html)

<a id="ref-lattneragree"></a>
- **[LattnerAgree]** Chris Lattner, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/129891.html)

<a id="ref-lattnerfunction"></a>
- **[LattnerFunction]** Chris Lattner, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130188.html)

<a id="ref-lattnerlower"></a>
- **[LattnerLower]** Chris Lattner, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130629.html)

<a id="ref-lattnertransition"></a>
- **[LattnerTransition]** Chris Lattner, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130355.html)

<a id="ref-malyutindistinguish"></a>
- **[MalyutinDistinguish]** Danila Malyutin, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130320.html)

<a id="ref-parzyszekacronym"></a>
- **[ParzyszekAcronym]** Krzysztof Parzyszek, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130306.html)

<a id="ref-parzyszekacronym2"></a>
- **[ParzyszekAcronym2]** Krzysztof Parzyszek, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130323.html)

<a id="ref-parzyszekdistinguish"></a>
- **[ParzyszekDistinguish]** Krzysztof Parzyszek, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130282.html)

<a id="ref-picusacronym"></a>
- **[PicusAcronym]** Alon Picus, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130321.html)

<a id="ref-reamesconcern"></a>
- **[ReamesConcern]** Philip Reames, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130181.html)

<a id="ref-ricciacronyms"></a>
- **[RicciAcronyms]** Bruno Ricci, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130328.html)

<a id="ref-robinsonagree"></a>
- **[RobinsonAgree]** Paul Robinson, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130111.html)

<a id="ref-robinsondistinguish"></a>
- **[RobinsonDistinguish]** Paul Robinson, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/129920.html)

<a id="ref-robinsondistinguish2"></a>
- **[RobinsonDistinguish2]** Paul Robinson, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130229.html)

<a id="ref-robinsontransition"></a>
- **[RobinsonTransition]** Paul Robinson, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130397.html)

<a id="ref-turnercamelback"></a>
- **[TurnerCamelBack]** Zachary Turner, [Phabricator review comment](https://reviews.llvm.org/D57896#1402264)

<a id="ref-turnerlldb"></a>
- **[TurnerLLDB]** Zachary Turner, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130213.html)

<a id="ref-ueyama"></a>
- **[Ueyama]** Rui Ueyama, [llvm-dev post](http://lists.llvm.org/pipermail/llvm-dev/2019-February/130435.html)

