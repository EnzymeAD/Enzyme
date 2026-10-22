if(ENZYME_FLANG)
  set(ENZYME_FLANG_ENABLED 1)
else()
  set(ENZYME_FLANG_ENABLED 0)
endif()

# Locate an MPI implementation for the Fortran MPI regression tests
# (enzyme/test/Fortran/mpi); the tests are marked unsupported if none is
# found. The include and link flags are serialized into lit substitutions
# %mpi_include and %mpi_libs.
set(ENZYME_MPI_FORTRAN_ENABLED 0)
set(ENZYME_MPI_INCLUDE_FLAGS "")
set(ENZYME_MPI_LINK_FLAGS "")
if(ENZYME_FORTRAN)
  # enable_language(Fortran) in ../Fortran only covers that directory scope.
  enable_language(Fortran)
  find_package(MPI COMPONENTS Fortran QUIET)
  if(MPI_Fortran_FOUND)
    set(ENZYME_MPI_FORTRAN_ENABLED 1)
    foreach(dir IN LISTS MPI_Fortran_INCLUDE_DIRS MPI_Fortran_MODULE_DIR)
      string(APPEND ENZYME_MPI_INCLUDE_FLAGS " -I${dir}")
    endforeach()
    foreach(dir IN LISTS MPI_Fortran_LIBRARY_DIRS)
      string(APPEND ENZYME_MPI_LINK_FLAGS " -L${dir}")
    endforeach()
    if(MPI_Fortran_LINK_FLAGS)
      string(APPEND ENZYME_MPI_LINK_FLAGS " ${MPI_Fortran_LINK_FLAGS}")
    endif()
    foreach(lib IN LISTS MPI_Fortran_LIBRARIES)
      if(EXISTS "${lib}")
        string(APPEND ENZYME_MPI_LINK_FLAGS " ${lib}")
      else()
        string(APPEND ENZYME_MPI_LINK_FLAGS " -l${lib}")
      endif()
    endforeach()
    message(
      STATUS
        "Enzyme Fortran MPI tests enabled: ${ENZYME_MPI_INCLUDE_FLAGS} / ${ENZYME_MPI_LINK_FLAGS}")
  else()
    message(
      STATUS "MPI Fortran bindings not found: MPI Fortran tests unsupported")
  endif()
endif()

# Locate SUNDIALS with its Fortran module interface for the CVODE tests
# (enzyme/test/Integration/Fortran/cvode); they are unsupported without it.
# Point SUNDIALS_DIR at the lib/cmake/sundials directory of an installation
# configured with BUILD_FORTRAN_MODULE_INTERFACE=ON by the same Fortran
# compiler (.mod files are compiler specific). The include and link flags are
# serialized into the lit substitutions %cvode_include and %cvode_libs.
set(ENZYME_CVODE_FORTRAN_ENABLED 0)
set(ENZYME_CVODE_INCLUDE_FLAGS "")
set(ENZYME_CVODE_LINK_FLAGS "")
if(ENZYME_FORTRAN)
  find_package(SUNDIALS CONFIG QUIET)
  if(SUNDIALS_FOUND AND TARGET SUNDIALS::fcvode_mod)
    # SUNDIALS_DIR is <libdir>/cmake/sundials; the modules are in <prefix>/fortran.
    get_filename_component(_enzyme_sundials_libdir "${SUNDIALS_DIR}/../.." ABSOLUTE)
    get_filename_component(_enzyme_sundials_prefix "${_enzyme_sundials_libdir}/.." ABSOLUTE)
    if(EXISTS "${_enzyme_sundials_prefix}/fortran/fcvode_mod.mod")
      set(ENZYME_CVODE_FORTRAN_ENABLED 1)
      set(ENZYME_CVODE_INCLUDE_FLAGS "-I${_enzyme_sundials_prefix}/fortran")
      set(ENZYME_CVODE_LINK_FLAGS
          "-L${_enzyme_sundials_libdir} -Wl,-rpath,${_enzyme_sundials_libdir}")
      foreach(lib fcvode_mod cvode fnvecserial_mod nvecserial fsunlinsolspgmr_mod
                  sunlinsolspgmr fsunlinsoldense_mod sunlinsoldense fsunmatrixdense_mod
                  sunmatrixdense fcore_mod core)
        string(APPEND ENZYME_CVODE_LINK_FLAGS " -lsundials_${lib}")
      endforeach()
      message(
        STATUS
          "Enzyme Fortran CVODE tests enabled: ${ENZYME_CVODE_INCLUDE_FLAGS} / ${ENZYME_CVODE_LINK_FLAGS}")
    endif()
  endif()
  if(NOT ENZYME_CVODE_FORTRAN_ENABLED)
    message(
      STATUS "SUNDIALS Fortran modules not found: CVODE Fortran tests unsupported")
  endif()
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(ENZYME_TEST_DEPS LLVMEnzyme-${LLVM_VERSION_MAJOR})

add_subdirectory(ActivityAnalysis)
add_subdirectory(TypeAnalysis)
add_subdirectory(Enzyme)
if(ENZYME_ENABLE_PLUGINS)
  if(${Clang_FOUND})
    add_subdirectory(Integration)
    # fmt: off
    if (${LLVM_VERSION_MAJOR} GREATER_EQUAL 11)
      add_custom_target(test-cmake COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && mkdir ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && cd ${CMAKE_CURRENT_BINARY_DIR}/test_cmake &&  ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_BINARY_DIR}/test_cmake ${CMAKE_CURRENT_SOURCE_DIR}/test_find_package -DEnzyme_DIR=${CMAKE_BINARY_DIR} -DCMAKE_C_COMPILER=$<TARGET_FILE:clang> && ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/test_cmake DEPENDS ClangEnzymeFlags LLDEnzymeFlags)
    else()
      add_custom_target(test-cmake COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && mkdir ${CMAKE_CURRENT_BINARY_DIR}/test_cmake && cd ${CMAKE_CURRENT_BINARY_DIR}/test_cmake &&  ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_BINARY_DIR}/test_cmake ${CMAKE_CURRENT_SOURCE_DIR}/test_find_package -DEnzyme_DIR=${CMAKE_BINARY_DIR} -DCMAKE_C_COMPILER=$<TARGET_FILE:clang> && ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/test_cmake DEPENDS ClangEnzymeFlags)
    endif()
    # fmt: on
  endif()
endif()
if(ENZYME_FORTRAN)
  list(APPEND ENZYME_TEST_DEPS EnzymeFortran)
  if(ENZYME_FLANG)
    list(APPEND ENZYME_TEST_DEPS FlangEnzyme-${LLVM_VERSION_MAJOR}
         LLDEnzyme-${LLVM_VERSION_MAJOR})
  endif()
  add_subdirectory(Fortran)
  # Added here rather than from Integration/CMakeLists.txt, which is only
  # reached when clang is found.
  add_subdirectory(Integration/Fortran)
endif()
if(ENZYME_BC_LOADER)
  add_subdirectory(BCLoader)
endif()
if(ENZYME_MLIR)
  add_subdirectory(MLIR)
endif()
