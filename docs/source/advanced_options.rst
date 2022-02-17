Advanced options
================

Enzyme has several advanced options that may be of interest.

Performance options
-------------------

Disabling Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~

The ``enzyme-preopt`` option disables the preprocessing optimizations run by the Enzyme pass, except for the absolute minimum neccessary.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-preopt=1
    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-preopt=0


Forced Inlining
~~~~~~~~~~~~~~~

The ``enzyme-inline`` option forcibly inlines all subfunction calls. The ``enzyme-inline-count`` option limits the number of calls inlined by this utility.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-inline=1
    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-inline=1 -enzyme-inline-count=100


Compressed Bool Cache
~~~~~~~~~~~~~~~~~~~~~

The ``enzyme-smallbool`` option allows Enzyme's cache to store 8 boolean (i1) values inside a single byte rather than one value per byte.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-smallbool=1


Semantic options
----------------

Loose type analysis
~~~~~~~~~~~~~~~~~~~

The ``enzyme-loose-types`` option tells Enzyme to make an educated guess about the type of a value it cannot prove, rather than emit a compile-time error and fail. This can be helpful for starting to bootstrap code with Enzyme but shouldn't be used in production as Enzyme may make an incorrect guess and create an incorrect gradient.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-loose-types=1


Assume inactivity of undefined functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``enzyme-emptyfn-inactive`` option tells activity analysis to assume that all calls to functions whose definitions aren't available and aren't explicitly given a custom gradient via metadata are assumed to be inactive. This can be useful for assuming printing functions don't impact derivative computations and provide a performance benefit, as well as getting around a compile-time error where the derivative of a foreign function is not known. However, this option should be used carefully as it may result in incorrect behavior if it is used to incorrectly assume a call to a foreign function doesn't impact  the derivative computation. As a result, the recommended way to remedy this is to mark the function as inactive explicitly, or provide a custom gradient via metadata.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-emptyfn-inactive=1


Assume inactivity of unmarked globals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``enzyme-globals-default-inactive`` option tells activity analysis to assume that global variables without an explicitly defined shadow global are assumed to be inactive. Like `enzyme_emptyfnconst`, this option should be used carefully as it may result in incorrect behavior if it is used to incorrectly assume that a global variable doesn't contain data used in a derivative computation.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-globals-default-inactive=1

Cache behavior
~~~~~~~~~~~~~~

The ``enzyme-cache-never`` option tells the cache to recompute all load values, even if alias analysis isn't able to prove the legality of such a recomputation. This may improve performance but is likely to result in incorrect derivatives being produced as this is not generally true.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-cache-never=1

In contrast, the `enzyme-cache-always` option tells the cache to still cache values that alias analysis and differential use analysis say are not needed to be cached (perhaps being legal to recompute instead). This will usually decrease performance and is intended for developers in order to catch caching bugs.::
    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-cache-always=1


Debugging options for developers
--------------------------------

enzyme-print
~~~~~~~~~~~~

This option prints out functions being differentiated before preprocessing optimizations, after preprocessing optimizations, and after being synthesized by Enzyme. It is mostly use to debug the AD process.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-print
    prefn:

    ; Function Attrs: norecurse nounwind readnone uwtable
    define double @square(double %x) #0 {
    entry:
      %mul = fmul double %x, %x
      ret double %mul
    }

enzyme-print-activity
~~~~~~~~~~~~~~~~~~~~~

This option prints out the results of activity analysis as they are being derived. The output is somewaht specific to the analysis pass and is only intended for developers.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-print-activity
    in new function diffesquare nonconstant arg double %0
    VALUE nonconst from arg nonconst double %x
    checking if is constant[3]   %mul = fmul double %x, %x
    < UPSEARCH3>  %mul = fmul double %x, %x
    VALUE nonconst from arg nonconst double %x
    nonconstant(3)  up-inst   %mul = fmul double %x, %x op double %x
    </UPSEARCH3>  %mul = fmul double %x, %x
    couldnt decide nonconstants(3):  %mul = fmul double %x, %x
    Value nonconstant (couldn't disprove)[3]  %mul = fmul double %x, %x

enzyme-print-type
~~~~~~~~~~~~~~~~~

This option prints out the results of type analysis as they are being derived. The output is somewaht specific to the analysis pass and is only intended for developers.::

    $ opt input.ll -load=./Enzyme/LLVMEnzyme-7.so -enzyme -enzyme-print-type
    analyzing function square
     + knowndata: double %x : {[-1]:Float@double} - {}
     + retdata: {}
    updating analysis of val: double %x current: {} new {[-1]:Float@double}
    updating analysis of val: double %x current: {[-1]:Float@double} new {[-1]:Float@double} from double %x
    updating analysis of val:   %mul = fmul double %x, %x current: {} new {}
    updating analysis of val: double %x current: {[-1]:Float@double} new {[-1]:Float@double} from   %mul = fmul double %x, %x
    updating analysis of val: double %x current: {[-1]:Float@double} new {[-1]:Float@double} from   %mul = fmul double %x, %x
    updating analysis of val:   %mul = fmul double %x, %x current: {} new {[-1]:Float@double} from   %mul = fmul double %x, %x
