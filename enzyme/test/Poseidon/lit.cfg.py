# -*- Python -*-

import os
import platform

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

config.available_features.add('poseidon')
# Before LLVM 21, SCEV does not carry no-wrap through the trunc of the i64
# canonical IV, so B's address in an i32-indexed reduction is no add-recurrence
# and the scalar-loop matmul recognizer finds nothing.
if int(config.llvm_ver) >= 21:
    config.available_features.add('scev-trunc-iv-nowrap')
# Up to LLVM 23 the code generator of a -ffp-contract=fast unit fuses
# floating-point operations that carry no 'contract' flag as well.
if int(config.llvm_ver) < 24:
    config.available_features.add('global-fp-contract')
