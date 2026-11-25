# -*- Python -*-

import os
import platform
import subprocess
import sys

import lit
import lit.formats

config.name = 'Poseidon'
# RUN lines use subshells, which lit's internal shell does not support.
LLVM_VERSION_MAJOR = lit.__versioninfo__[0]
if LLVM_VERSION_MAJOR >= 23:
    config.test_format = lit.formats.ShTest(platform.system() != 'Windows',
                                            force_execute_external=True)
else:
    config.test_format = lit.formats.ShTest(platform.system() != 'Windows')
config.suffixes = ['.ll', '.c', '.cpp', '.cu', '.hip']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.poseidon_obj_root
config.excludes = ['Inputs', 'CMakeFiles']

config.environment['PATH'] = os.path.pathsep.join(
    [config.llvm_tools_dir, config.environment['PATH']])
config.environment['LD_LIBRARY_PATH'] = os.path.pathsep.join(
    [config.llvm_libs_dir, config.environment.get('LD_LIBRARY_PATH', '')])
for var in ('CUDA_VISIBLE_DEVICES', 'HOME'):
    if var in os.environ:
        config.environment[var] = os.environ[var]
# A compile that names no cost model looks one up in the user cache: the tests
# get one of their own, which holds none, never the developer's.
config.environment['XDG_CACHE_HOME'] = os.path.join(config.test_exec_root,
                                                    'xdg-cache')

poseidon_bin = os.path.join(config.poseidon_build_root, 'bin')
poseidon_lib = os.path.join(config.poseidon_build_root, 'lib')
poseidon_runtime = os.path.join(config.poseidon_src_root, 'runtime')
poseidon_plugin = os.path.join(
    poseidon_lib, 'Poseidon-' + config.llvm_ver + config.llvm_shlib_ext)
python = '"' + sys.executable + '"'

# Loading a plugin. Before LLVM 22 clang parses -mllvm before loading
# -fpass-plugin, and from LLVM 24 opt loads -load-pass-plugin only after
# parsing its command line, so a plugin's options exist only if it is also
# loaded with -load. A compile that loads Poseidon's plugin defines
# POSEIDON_DEVICE_MATH, as poseidon-clang does.
def clang_load(plugin):
    if int(config.llvm_ver) < 22:
        return '-fpass-plugin=' + plugin + ' -Xclang -load -Xclang ' + plugin
    return '-fpass-plugin=' + plugin
def opt_load(plugin):
    return '-load-pass-plugin=' + plugin + ' -load=' + plugin
clang_load_poseidon = clang_load(poseidon_plugin) + ' -DPOSEIDON_DEVICE_MATH'

# lit applies substitutions in list order, so a name that is a prefix of
# another (%loadClangPoseidon, %poseidon_clang, %cmake) comes after it.
config.substitutions += [
    # The plugins, and Enzyme's ('Enzyme' last: it differentiates the probes).
    ('%loadClangPoseidonEnzyme',
     clang_load_poseidon + ' ' + clang_load(config.enzyme_clang_plugin)),
    ('%loadClangPoseidon', clang_load_poseidon),
    ('%loadClangEnzyme', clang_load(config.enzyme_clang_plugin)),
    ('%loadPoseidonEnzyme',
     opt_load(poseidon_plugin) + ' ' + opt_load(config.enzyme_opt_plugin)),
    ('%loadPoseidon', opt_load(poseidon_plugin)),
    ('%poseidon_plugin', poseidon_plugin),
    ('%enzyme_clang_plugin', config.enzyme_clang_plugin),
    # The tools.
    ('%poseidon_clangxx', os.path.join(poseidon_bin, 'poseidon-clang++')),
    ('%poseidon_clang', os.path.join(poseidon_bin, 'poseidon-clang')),
    ('%poseidon_calibrate', os.path.join(poseidon_bin, 'poseidon-calibrate')),
    ('%poseidon_probe', os.path.join(poseidon_bin, 'poseidon-probe')),
    ('%python', python),
    ('%poseidon_explain', python + ' ' + os.path.join(
        config.poseidon_src_root, 'tools', 'explain', 'poseidon-explain')),
    ('%herbie_binary', config.poseidon_herbie_binary),
    # The programs and opt plugins built for the tests, and the made-up cost
    # models the tests price from.
    ('%poseidon_test_programs', config.poseidon_test_programs),
    ('%gpu_cost_model',
     os.path.join(config.poseidon_cost_models, 'cm_gpu_synthetic.csv')),
    ('%host_cost_model',
     os.path.join(config.poseidon_cost_models, 'cm_cpu_host.csv')),
    ('%cost_models', config.poseidon_cost_models),
    # LLVM, as in Enzyme's suites; %clang++ is %clang followed by ++.
    ('%shlibext', config.llvm_shlib_ext),
    ('%opt', config.llvm_tools_dir + '/opt'),
    ('%clang', config.llvm_tools_dir + '/clang'),
    # The CMake package of this build tree, and what installing it takes.
    ('%cmake_generator', '-G "' + config.cmake_generator +
     '" -DCMAKE_MAKE_PROGRAM=' + config.cmake_make_program),
    ('%cmake', '"' + config.cmake + '"'),
    ('%poseidon_package_dir', config.poseidon_build_root),
    ('%install_libdir', config.install_libdir),
    ('%enzyme_binary_dir', config.enzyme_obj_root),
    # The runtimes, for tests that link them themselves: the profiler, and the
    # GEMM runtimes for tests that call an entry point instead of having the
    # pass emit the call.
    ('%poseidon_profile_lib',
     os.path.join(poseidon_lib, 'libposeidon_profile.a')),
    ('%poseidon_profile_inc', os.path.join(poseidon_runtime, 'fpprofiler') +
     ' -I' + os.path.join(config.enzyme_src_root, 'include')),
    ('%poseidon_runtime_inc', poseidon_runtime),
    # The CUDA and the HIP builds of the GPU runtimes, side by side.
    ('%gpu_arch', config.gpu_arch),
    ('%amdgpu_arch', config.amdgpu_arch),
    ('%cuda_lib_dir', config.cuda_lib_dir),
    ('%rocm_path', config.rocm_path),
    ('%poseidon_profile_cuda_lib',
     os.path.join(poseidon_lib, 'libposeidon_profile_cuda.a')),
    ('%poseidon_rt_lib',
     os.path.join(poseidon_lib, 'libposeidon_rt.a') + config.poseidon_rt_link),
    ('%poseidon_rt_hip_lib', os.path.join(poseidon_lib, 'libposeidon_rt_hip.a')),
]

# Up to LLVM 23 the code generator of a -ffp-contract=fast unit fuses
# floating-point operations that carry no 'contract' flag as well.
if int(config.llvm_ver) < 24:
    config.available_features.add('global-fp-contract')
# From LLVM 23 clang compiles CUDA and HIP through the new offload driver, in
# which a HIP compile's device half is an LTO pre-link compile.
if int(config.llvm_ver) >= 23:
    config.available_features.add('offload-new-driver')
# From LLVM 23 a function's GUID is kept as !guid metadata.
if int(config.llvm_ver) >= 23:
    config.available_features.add('guid-metadata')
# Tests that generate code for a target, or whose passes read its target
# information, as Enzyme's suites name them.
for arch in config.llvm_targets_to_build.split(';'):
    config.available_features.add(arch.lower() + '-registered-target')
if os.path.isfile(config.enzyme_clang_plugin):
    config.available_features.add('enzyme')
if os.path.isfile(config.poseidon_herbie_binary) and os.access(
        config.poseidon_herbie_binary, os.X_OK):
    config.available_features.add('herbie')
# poseidon-calibrate's CPU microbenchmark JIT-compiles with llvmlite.
if subprocess.run([config.poseidon_calibrate_python, '-c',
                   'import llvmlite.binding'],
                  capture_output=True).returncode == 0:
    config.available_features.add('cpu-calibration')
# The CUDA and the HIP builds of the GPU runtimes: the runtime libraries and
# the device profiler the plugin carries (profile generation for a GPU without
# one stops at its first probe), and a device to run what links them. HIP
# compiles also need the device libraries of the ROCm installation.
if config.poseidon_has_cuda:
    config.available_features.add('cuda-runtime')
    config.available_features.add('nvptx-device-profiler')
if config.poseidon_has_hip:
    config.available_features.add('amdgpu-device-profiler')
    if os.path.exists('/dev/kfd'):
        config.available_features.add('hip-runtime')
if config.rocm_path and os.path.isdir(
        os.path.join(config.rocm_path, 'amdgcn', 'bitcode')):
    config.available_features.add('rocm')
# The error-corrected GEMM runtime, which only a build with cuMpSGEMM has.
if config.poseidon_has_tcec:
    config.available_features.add('tcec-cumpsgemm')
