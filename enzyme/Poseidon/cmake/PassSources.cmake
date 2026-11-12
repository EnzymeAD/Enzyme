# poseidon_pass_sources(<out_var> <pass_dir> <binary_dir> [CONFIGURE_DEPENDS])
#
# The sources of the pass in <pass_dir>, relative to it and sorted: every *.cpp
# and *.h at any depth, except under the directories that hold no part of the
# pass (the runtimes, the tools, the CMake scripts), under any test or build
# directory, and under <binary_dir> if it lies inside <pass_dir>.
# runtime/ozaki/OzakiII.h and runtime/tcec/tcec_scale.h are sources of the pass
# although they lie with the runtime: the pass prices with their tables and
# simulates with their scaling. CONFIGURE_DEPENDS globs again
# at build time, so that a file added or removed reconfigures.
function(poseidon_pass_sources out_var pass_dir binary_dir)
  file(
    GLOB_RECURSE
    _all
    RELATIVE
    ${pass_dir}
    ${ARGN}
    ${pass_dir}/*.cpp
    ${pass_dir}/*.h)
  set(_binary "")
  if(binary_dir)
    file(RELATIVE_PATH _binary ${pass_dir} ${binary_dir})
    if(_binary MATCHES "^\\.\\.(/|$)" OR IS_ABSOLUTE "${_binary}")
      set(_binary "")
    endif()
  endif()
  set(_sources "")
  foreach(_file IN LISTS _all)
    if(
      (_file MATCHES "^(runtime|tools|cmake)/" AND
        NOT _file MATCHES "^runtime/(ozaki/OzakiII|tcec/tcec_scale)\\.h$")
      OR _file MATCHES "(^|/)(test|tests|unittests|CMakeFiles|build[^/]*)/")
      continue()
    endif()
    if(_binary)
      string(FIND "${_file}" "${_binary}/" _at)
      if(_at EQUAL 0)
        continue()
      endif()
    endif()
    list(APPEND _sources ${_file})
  endforeach()
  list(SORT _sources COMPARE STRING)
  set(${out_var} ${_sources} PARENT_SCOPE)
endfunction()
