# The cost model

Poseidon prices every candidate from one CSV measured on the device the code
will run on. No model ships with Poseidon: `poseidon-calibrate` produces one,
and a compile that is not told which model to use looks one up by the module's
target-cpu:

    -poseidon-cost-model=<csv>              what the compile was told
    POSEIDON_COST_MODEL=<csv>               what the environment says
    $XDG_CACHE_HOME/poseidon/cm_<arch>_*.csv  what this user has measured
                                            (default ~/.cache/poseidon)

No match stops the compile naming `poseidon-calibrate`; more than one stops it
naming the files it found. A model whose `# native_arch=` header does not name
the compile target aborts as well: another device's CSV is that device's
prices, and would reproduce that device's picks. Results solved with a measured
model are reproduced from that model, so it is kept with them and named with
`-poseidon-cost-model`.

Herbie ranks the rewrites its algebraic search finds by a platform cost table.
A compile passes the platform that sits next to the CSV it prices from
(`<csv>.herbie.rkt`), so the search and the DP describe one device.

## Measuring a device

    poseidon-calibrate --gpu 0                 # writes ~/.cache/poseidon/cm_<arch>_<gpu>.csv
    poseidon-calibrate --gpu 0 --out <dir>     # or somewhere else
    poseidon-calibrate --gpu 0 --check         # is the resolved model this device's?
    poseidon-calibrate --gpu 0 --product MxNxK:bcolmajor  # plus the rows of an M x N x K product of constant extents
    poseidon-calibrate --gpu 0 --ozaki-rect MxK      # plus the rows of an M x K operand at run-time column counts
    poseidon-calibrate --cpu native            # this host, for a compile with -march=native
    poseidon-calibrate --cpu x86-64            # this host, for a compile with -march=x86-64, the default on x86-64
    poseidon-calibrate --cpu x86-64 --check

A run that was asked for a row and could not measure it, a class that did not
build or run on the device for instance, writes the rows it measured, names the
others on its last line and exits with status 1; the same command run again
measures only those. The GPU is the one the CUDA microbenchmark finds, else
the HIP one; a microbenchmark that cannot initialize the device, or cannot be
started, stops the run with its reason.

A host CPU is measured for the target-cpu its compiles use: `--cpu` takes the
name `-march=` takes, and `native` stands for the CPU LLVM detects on this
machine, whose features the measurement takes as well. The model is
`cm_<target-cpu>_<cpu>.csv`, in the same place as a GPU's. Its rows are timed
by `ops_cpu.py`, which JIT-compiles each microbenchmark with llvmlite for
that CPU and stops if llvmlite's LLVM does not know it. A trial run sizes each
loop to run for a quarter of a second, and a row is the time of one operation
less that of the same loop without it, in the unit of the GPU rows, clock
cycles times two, so a model takes a few minutes and does not depend on the
trip count. A host model holds rows of two kinds, which mean different things:

- An arithmetic operation, comparison, conversion, fused multiply-add, square
  root, minimum, maximum, rounding or sign operation, or `powi` row is the
  operation's latency: a dependent chain through the operation, as the GPU
  microbenchmark chains it, or, where the compiler would fold that chain, the
  operation on fixed inputs that a restoring step makes depend on the
  previous result.
- A math-library row is the function's reciprocal throughput: independent
  calls on fixed inputs. A dependent chain would price a call whose result
  follows a predicted branch (an overflowing `exp`, a saturated `tanh`) at
  nothing. The inputs lie inside the function's domain, on the ordinary path
  of its routine in the C library, in the intervals `ops_cpu.py` lists
  beside each function.

The GPU rows are reciprocal throughputs throughout, so a GPU model and a host
model are not comparable row for row. Before a loop is timed, the script
checks its machine code for an instruction of every operation it times, and
stops naming the row if the compiler has folded one away. The script runs under
the Python 3 the build was configured with (`Python3_EXECUTABLE`) or the one
`--python` names, which needs the `llvmlite` package. A host model holds the
`float` and `double` per-op rows, the formats a precision change offers a host
function, and no row of the arms that measure a GPU: `--cpu` refuses those
arms and `--gpu`.

Seven arms write disjoint row families into the same file; `--only` runs a
subset, `--force` re-measures a family that is already there. An eighth,
`herbie-platform`, measures nothing: it translates the CSV into
`<csv>.herbie.rkt` beside it. It runs after `ops` and in a full run, and
because it needs no device, `--only herbie-platform --out <csv>` also runs on a
machine that does not have the GPU the CSV was measured on.

| Arm | Rows | What is measured |
|---|---|---|
| ops | `<op>,<type>,<cost>`, `wmma_mma_<shape>,<in>_<acc>,<rel>` | Saturated reciprocal throughput per (op, type), from dependent op chains on every SM, or on a host from JIT-compiled loops. On NVIDIA a tensor-core tile row exists when a real tiled WMMA GEMM of that scheme beats a scalar-FP64 GEMM of the same shape, and holds the ratio of their times; a compile reads only whether the row exists. Also writes the `# native_arch=` and `# scalar_types=` headers. |
| ozaki | `library_ozaki_rel,nm<8..14>`, `library_ozaki_rel,dgemm` | library-call Ozaki-II GEMM per modulus count, and the native cuBLAS DGEMM. |
| ozaki-rect | `library_ozaki_rect_rel,nm<8..14>_m<M>n<N>k<K>`, `library_ozaki_rect_rel,dgemm_m<M>n<N>k<K>`, `library_ozaki_rect_rel,nm<8..14>_m<M>k<K>` | Host dispatches at the `--product` and `--ozaki-rect` shapes, per MAC, after a warm-up call. A product `MxNxK` measures the Ozaki-II library call and the native DGEMM of a call with exactly those extents: a call with few columns does not amortize the conversion of its M x K operand, and a DGEMM with few columns runs far below the square rate. A shape `MxK` measures the Ozaki-II library call at 12288 columns, assuming a cost linear in the column count; it stands for a call whose column count is a run-time value, and is dearer per MAC for few columns than it says. The shapes are the ones the calls run at: a call whose three extents are constants of its compile and not a square is priced from the rows of its own shape, a square call from the square rows; a compile without them warns with the `--product` argument that measures them, and prices Ozaki-II meanwhile as the square product the call embeds in (the square row times the padding ratio) and the native DGEMM from the square row. The shape of a call whose extents are run-time values is not known to the compile: its Ozaki-II candidates are priced from the `MxK` rows of its profiled (M, K), named by `--ozaki-rect`, its native DGEMM from the square row. Rows of shapes a run does not name are kept. |
| tcec | `library_tcec_rel,fp16tcec` | library-call error-corrected GEMM, on the cuMpSGEMM the compiler is built with. A compiler built without cuMpSGEMM has no such dispatch, and the arm says so and writes no row. |
| direct | `library_direct_rel,<in>_<acc>` | library-call reduced-precision cuBLAS GEMM. |
| inkernel | `inkernel_rel,<class>`, `inkernel_rel,<class>_m<M>n<N>k<K>_<layout>` | Tensor-core raises inside a kernel, one class at a time: the harness `inkernel.cu` is compiled through `poseidon-clang++` with `-poseidon-pin`, so what is timed is the code a real solve emits. The class row is measured on a square product of `--n` (2048) with B strided along K; a product `MxNxK:<layout>` measures the rows of exactly those extents with B strided (`browmajor`) or contiguous (`bcolmajor`) along K. A product with few output columns or B contiguous along K does not run at the class row's rate: the raise of a product whose extents are constants of its compile is priced from the rows of its extents and layout, and a compile without them warns with the `--product` argument that measures them and prices it meanwhile from the class row. A product whose output extents are run-time values is priced from the class row. |
| operand-min | `inkernel_operand_min,<class>`, `library_direct_operand_min,<in>_<acc>`, `library_tcec_operand_min,fp16tcec` | Per product class whose operands are FP16, BF16 or TF32, the smallest operand magnitude its product keeps on this device, by the code the class's own arm times and at the same shape (`operand_min.h`): the in-kernel classes the pass proposes on the square harness of `--n`, and the library calls the model prices (its `library_direct_rel` classes, and `library_tcec_rel,fp16tcec`) by their calibrator. Each binade from the format's smallest subnormal to ten above its smallest normal is put through the product in one operand, the other a sparse matrix of plus and minus 2^s, and compared bit for bit with the exact result. 2^s keeps every partial sum at the accumulator's smallest normal or above, so what is measured is the operand and not an accumulator that flushes its own subnormals. The row is the lowest binade from which every binade up is kept in both operands. A library or matrix unit that flushes subnormal inputs, on the kernels it picks for products of a realistic size, measures the format's smallest normal. A class whose floor is not established (its top binade is lost) gets no row and fails the run. The accuracy model reads an operand below the row as zero, and does not offer a candidate whose profiled operands reach below it, as it does not offer one whose format flushes them. A model with the rows of some classes does not propose a class whose row it lacks, and warns as for a missing price; a model with none is priced with the formats' own subnormals, which a solve under `-poseidon-print` names. |
| herbie-platform | none (writes `<csv>.herbie.rkt`) | Nothing. The per-op rows are translated into the Herbie platform the algebraic search ranks candidates by, so Herbie and the DP price the same device. A compile against a GPU CSV with no platform beside it aborts rather than search under another architecture's costs, and Herbie result caches are keyed by the platform name so one device's rewrites are never replayed for another. |

An AMD GPU is measured by the HIP builds of the same tools
(`poseidon-calibrate-ops-hip` and `poseidon-calibrate-library-hip`, built when the
configuration names `POSEIDON_ROCM_PATH`, for the processor
`POSEIDON_AMDGPU_ARCH` names, default `gfx942`), which `poseidon-calibrate`
runs when the CUDA build finds no device with that id; the device is the one
`HIP_VISIBLE_DEVICES` (or `--gpu`) selects.
They write `cm_gfx<NNN>_<device>.csv` with the same row names in the same unit,
and a compile for that processor finds it by its triple. Rows the device has no
runtime or instruction for are absent, so the solver does not propose those
classes: the WMMA tile rows and the in-kernel raises (NVPTX-only), the
error-corrected library call (no HIP runtime) and `library_direct_rel,tf32_f32`
(hipBLAS has no TF32 compute type), with their `*_operand_min` rows. Its Herbie
platform is `hip-gfx<NNN>`.

Every `*_rel` row is one number in one unit: the candidate's wall clock over the
scalar-FP64 GEMM's, at the same shape on the same device, so the DP compares an
in-kernel raise against a library call directly. The in-kernel arm learns
which candidates the pass proposes from the explanation (`-poseidon-explain`)
of a compile of its harness (installed under `share/poseidon/tools/calibrate`),
whose profile it records itself at extents of at most 128 with B in the layout
it measures; the profile decides which candidates exist, never what they
measure. A class with no measured row is not
proposed. The pass does not compose a price
for it out of per-MMA rows: a materialized raise also pays a scratch fill,
shared-memory staging and accumulator traffic that no per-MMA row contains.

## Format

One row per line, `opcode,type,cost`, plus `#`-prefixed headers. `native_arch`
is the architecture the file was measured on, a GPU's or the target-cpu of a
host; `scalar_types` lists the precisions the solver may propose on a GPU. A
row the compile needs, a header it reads, a line that is not a row, a row given
twice and a row of no family above each stop the compile, naming the model:
nothing is substituted for a part of it, and a misspelt row a candidate class
depends on does not silently withdraw the class. A new family is added to the
pass (`Model/CostTable.cpp`) with the arm that writes it. The DP cache is keyed
on a hash of the parsed rows, so changing a price invalidates a cached table
while reordering rows or editing comments does not.

Scalar costs are the precise, IEEE lowering of each operation (`div.rn.f32`,
libdevice `__nv_*` for the transcendentals; on AMD the IEEE `fdiv` lowering and
ROCm's ocml `__ocml_*`). Hardware `.approx` variants are 5x to 50x faster and
lose accuracy; a kernel that opts into them would need a parallel row family,
which does not exist. An NVIDIA GPU has no instruction for `fneg` or `fabs`:
either is an operand modifier of the instruction that reads the value, an
`FADD` when nothing else does, so the two rows are priced as `fadd`; on AMD
they are the 32-bit sign-bit operation the backend emits where it cannot fold
them. A row whose measurement is not positive stops the calibration, naming
the row.
