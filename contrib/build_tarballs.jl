using BinaryBuilder

sources = [
            joinpath(@__DIR__, "../../enzyme")
          ]

script = raw"""
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_TOOLCHAIN_FILE=/opt/$target/$target.toolchain \
    -DLLVM_DIR=${WORKSPACE}/destdir/lib/cmake/llvm
ninja install
"""

products(prefix) = [
    LibraryProduct(prefix, "LLVMEnzyme", :libenzyme),
]

dependencies = [
    BinaryBuilder.InlineBuildDependency(readchomp("build_LLVM.v6.0.1.jl"))]

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = [
    BinaryProvider.Linux(:i686, :glibc),
    BinaryProvider.Linux(:x86_64, :glibc),
    BinaryProvider.Linux(:x86_64, :musl),
    BinaryProvider.Linux(:aarch64, :glibc),
    BinaryProvider.Linux(:armv7l, :glibc),
    BinaryProvider.Linux(:powerpc64le, :glibc),
#    BinaryProvider.MacOS(),
#    BinaryProvider.Windows(:i686),
#    BinaryProvider.Windows(:x86_64),
#    BinaryProvider.FreeBSD(:x86_64),
]
platforms = expand_gcc_versions(platforms)

# Build 'em!
build_tarballs(
    ARGS,
    "enzyme",
    v"0.0.1",
    sources,
    script,
    platforms,
    products,
    dependencies,
)
