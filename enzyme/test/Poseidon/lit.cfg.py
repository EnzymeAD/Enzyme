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
config.suffixes = ['.ll', '.c', '.cpp', '.cu']
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

config.available_features.add('poseidon')
# Before LLVM 21, SCEV does not carry no-wrap through the trunc of the i64
# canonical IV, so B's address in an i32-indexed reduction is no add-recurrence
# and the scalar-loop matmul recognizer finds nothing.
if int(config.llvm_ver) >= 21:
    config.available_features.add('scev-trunc-iv-nowrap')
