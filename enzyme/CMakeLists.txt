cmake_minimum_required(VERSION 3.13)
project(Enzyme)

include(CMakePackageConfigHelpers)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)

set(ENZYME_MAJOR_VERSION 0)
set(ENZYME_MINOR_VERSION 0)
set(ENZYME_PATCH_VERSION 79)
set(ENZYME_VERSION
  ${ENZYME_MAJOR_VERSION}.${ENZYME_MINOR_VERSION}.${ENZYME_PATCH_VERSION})

add_definitions(-DENZYME_VERSION_MAJOR=${ENZYME_MAJOR_VERSION})
add_definitions(-DENZYME_VERSION_MINOR=${ENZYME_MINOR_VERSION})
add_definitions(-DENZYME_VERSION_PATCH=${ENZYME_PATCH_VERSION})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED ENZYME_CONFIGURED_WITH_PRESETS)
  set(CMAKE_CXX_FLAGS "-Wall -fno-rtti ${CMAKE_CXX_FLAGS} -Werror=unused-variable -Werror=dangling-else -Werror=unused-but-set-variable -Werror=return-type -Werror=nonnull -Werror=unused-result -Werror=reorder -Werror=switch")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -ggdb")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
  set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -ggdb -fno-omit-frame-pointer")
endif()

#SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -fno-omit-frame-pointer -fsanitize=address")
#SET(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENZYME_ENABLE_PLUGINS "Enable Clang/LLD/Opt plugins" ON)
option(ENZYME_BC_LOADER "Enable bitcode loader" ON)
option(ENZYME_CLANG "Build enzyme clang plugin" ON)
option(ENZYME_FLANG "Build enzyme flang symlink" OFF)
option(ENZYME_MLIR "Build enzyme mlir plugin" OFF)
option(ENZYME_IFX "Enable enzyme support for the Intel Fortran compiler IFX" OFF)
option(ENZYME_EXTERNAL_SHARED_LIB "Build external shared library" OFF)
option(ENZYME_STATIC_LIB "Build static library" OFF)
option(ENZYME_WARN_COMPILER "Warn if enzyme detects potentially incompatible compiler" ON)
set(ENZYME_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ENZYME_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${ENZYME_SOURCE_DIR}/cmake/modules")

set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")
if (APPLE)
    set(LLVM_SHLIBEXT ".dylib")
endif()
message( LLVM_SHLIBEXT = ${LLVM_SHLIBEXT} )

set(LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

get_filename_component(LLVM_ABSOLUTE_DIR
    "${LLVM_DIR}"
    REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

if (NOT EXISTS "${LLVM_DIR}")
  message("Looking for LLVM_DIR at ${LLVM_DIR}")
  message(SEND_ERROR "The given LLVM_DIR does not exist. Typo?")
endif()

set(LLVM_DIR "${LLVM_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)

if (EXISTS "${LLVM_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
  set(LLVM_DIR "${LLVM_DIR}/lib/cmake/llvm")
endif()

message("LLVM ABS DIR " ${LLVM_DIR})

message("CMAKE_PREFIX_PATH " ${CMAKE_PREFIX_PATH})

find_package(LLVM REQUIRED CONFIG)

if (NOT LLVM_ENABLE_PLUGINS)
  message("LLVM_ENABLE_PLUGINS OFF")
  message("Setting ENZYME_ENABLE_PLUGINS OFF")
  set(ENZYME_ENABLE_PLUGINS OFF)
endif()

if ((NOT ENZYME_EXTERNAL_SHARED_LIB) AND (NOT ENZYME_ENABLE_PLUGINS))
  message("ENZYME_EXTERNAL_SHARED_LIB OFF")
  message("ENZYME_ENABLE_PLUGINS OFF")
  message(SEND_ERROR "You have selected a build without creating the Enzyme plugin, nor Enzyme as a shared library -- and nothing will be built.")
endif()

if (NOT DEFINED LLVM_EXTERNAL_LIT)
  if(LLVM_DIR MATCHES ".*/cmake/llvm/?$")
      message("found llvm match ${CMAKE_MATCH_1} dir ${LLVM_DIR}")
    if (EXISTS ${LLVM_DIR}/../../../bin/llvm-lit)
      set(LLVM_EXTERNAL_LIT ${LLVM_DIR}/../../../bin/llvm-lit)
    else()
      set(LLVM_EXTERNAL_LIT lit)
    endif()
  else()
    if (EXISTS ${LLVM_DIR}/bin/llvm-lit)
      set(LLVM_EXTERNAL_LIT ${LLVM_DIR}/bin/llvm-lit)
    else()
      set(LLVM_EXTERNAL_LIT lit)
    endif()
  endif()
endif()

get_filename_component(LLVM_ABSOLUTE_LIT
                       "${LLVM_EXTERNAL_LIT}"
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

                   set(LLVM_EXTERNAL_LIT "${LLVM_ABSOLUTE_LIT}" CACHE FILEPATH "a" FORCE)
message("found llvm lit " ${LLVM_EXTERNAL_LIT})

list(INSERT CMAKE_PREFIX_PATH 0 "${LLVM_DIR}")
message("LLVM dir ${LLVM_DIR}")
if (ENZYME_CLANG)
if (DEFINED Clang_DIR)
    get_filename_component(Clang_ABSOLUTE_DIR
      "${Clang_DIR}"
    REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(Clang_DIR "${Clang_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)
  list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
  message("clang dir defined ${Clang_DIR}")
else()
  if(LLVM_DIR MATCHES ".*/cmake/llvm/?$")
    if (EXISTS ${LLVM_DIR}/../clang/../../libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR}/../clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/?$")
    if (EXISTS ${LLVM_DIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR}/lib/cmake/clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/cmake/?$")
      get_filename_component(LLDIR "${LLVM_DIR}" DIRECTORY)
    if (EXISTS ${LLDIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLDIR}/lib/cmake/clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  else()
    if (EXISTS ${LLVM_DIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR})
      list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  endif()
  message("clang dir from llvm ${Clang_DIR}")
endif()
else()
endif()

if (ENZYME_MLIR)
    if (DEFINED MLIR_DIR)
        get_filename_component(MLIR_ABSOLUTE_DIR
        "${MLIR_DIR}"
        REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
        set(MLIR_DIR "${MLIR_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)
        list(INSERT CMAKE_PREFIX_PATH 0 "${MLIR_DIR}")
        message("MLIR dir defined ${MLIR_DIR}")
    else()
      if(LLVM_DIR MATCHES ".*/cmake/llvm/?$")
        if (EXISTS ${LLVM_DIR}/../mlir/../../libMLIRIR.a)
            set(MLIR_DIR ${LLVM_DIR}/../mlir)
            list(INSERT CMAKE_PREFIX_PATH 0 "${MLIR_DIR}")
        else()
            message(SEND_ERROR "MLIR requested but not found")
        endif()
      elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/?$")
        if (EXISTS ${LLVM_DIR}/lib/libMLIRIR.a)
            set(MLIR_DIR ${LLVM_DIR}/lib/cmake/mlir)
          list(INSERT CMAKE_PREFIX_PATH 0 "${MLIR_DIR}")
        else()
            message(SEND_ERROR "MLIR requested but not found")
        endif()
      elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/cmake/?$")
          get_filename_component(LLDIR "${LLVM_DIR}" DIRECTORY)
        if (EXISTS ${LLDIR}/lib/libMLIRIR.a)
            set(MLIR_DIR ${LLDIR}/lib/cmake/mlir)
            list(INSERT CMAKE_PREFIX_PATH 0 "${MLIR_DIR}")
        else()
            message(SEND_ERROR "MLIR requested but not found")
        endif()
      else()
        if (EXISTS ${LLVM_DIR}/lib/libMLIRIR.a)
          set(MLIR_DIR ${LLVM_DIR})
          list(INSERT CMAKE_PREFIX_PATH 0 "${MLIR_DIR}")
        else()
            message(SEND_ERROR "MLIR requested but not found")
        endif()
      endif()
      message("mlir dir from llvm ${MLIR_DIR}")
    endif()
endif()

if(ENZYME_IFX)
    add_definitions(-DENZYME_IFX=1)
    message("Building with support for the Intel Fortran compiler IFX")
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

message("clang dir ${Clang_DIR}")
message("mlir dir ${MLIR_DIR}")

if (DEFINED Clang_DIR)
    find_package(Clang REQUIRED CONFIG PATHS ${Clang_DIR} NO_DEFAULT_PATH)
if (${Clang_FOUND})
    include_directories(${CLANG_INCLUDE_DIRS})
    message("clang inc dir ${CLANG_INCLUDE_DIRS}")
    add_definitions(${CLANG_DEFINITIONS})
endif()
else()
    set(Clang_FOUND 0)
endif()
message("found Clang ${Clang_FOUND}")

if (DEFINED MLIR_DIR)
    find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR} NO_DEFAULT_PATH)
    if (${MLIR_FOUND})
        include_directories(${MLIR_INCLUDE_DIRS})
        message("MLIR inc dir ${MLIR_INCLUDE_DIRS}")
        add_definitions(${MLIR_DEFINITIONS})
        add_definitions(-DENZYME_MLIR=1)
        list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
        include(TableGen)
        include(AddMLIR)
    endif()
else()
    set(MLIR_FOUND 0)
endif()
message("found MLIR ${MLIR_FOUND}")
# include(AddClang)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
message("LLVM_INSTALL_PREFIX: ${LLVM_INSTALL_PREFIX}")
message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message("found LLVM definitions " ${LLVM_DEFINITIONS})
message("found LLVM version " ${LLVM_VERSION_MAJOR})

if(LLVM_VERSION_MAJOR LESS 15)
  message(FATAL_ERROR "LLVM version < 15 is not supported")
endif()

option(ENZYME_FLANG_VERSION "Build for non-version compliant FLANG" OFF)
if (ENZYME_FLANG_VERSION)
  add_definitions(-DFLANG=1)
endif()

option(ENZYME_ROCM_VERSION "Build for non-version compliant ROCM" OFF)
if (ENZYME_ROCM_VERSION)
  add_definitions(-DROCM=1)
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/Enzyme)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")


list(GET LLVM_INCLUDE_DIRS 0 LLVM_IDIR)
message("first llvm include directory" ${LLVM_IDIR})

file(READ ${LLVM_IDIR}/llvm/Analysis/ScalarEvolution.h INPUT_TEXT)
string(REPLACE private public INPUT_TEXT "${INPUT_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolution.h" "${INPUT_TEXT}")


file(READ ${LLVM_IDIR}/llvm/Analysis/TargetLibraryInfo.h INPUT_TEXT)
string(REPLACE "class TargetLibraryInfo {" "class TargetLibraryInfo {public:" INPUT_TEXT "${INPUT_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/TargetLibraryInfo.h" "${INPUT_TEXT}")

file(READ ${LLVM_IDIR}/llvm/Transforms/Utils/ScalarEvolutionExpander.h INPUT_TEXT)

find_library(MPFR_LIB_PATH mpfr)
CHECK_INCLUDE_FILE("mpfr.h" HAS_MPFR_H)
message("MPFR lib: " ${MPFR_LIB_PATH})
message("MPFR header: " ${HAS_MPFR_H})

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolutionExpander.h" "${INPUT_TEXT}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

add_subdirectory(tools)
add_subdirectory(Enzyme)
if (ENZYME_BC_LOADER)
    add_subdirectory(BCLoad)
endif()
if (ENZYME_ENABLE_PLUGINS)
    add_subdirectory(test)
endif()

# Benchmarks have been moved to a separate repository to reduce repo size.
# Clone them alongside Enzyme to re-enable benchmarking:
# git clone https://github.com/EnzymeAD/enzyme-benchmarks benchmarks
if (ENZYME_ENABLE_PLUGINS AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"))
    add_subdirectory(benchmarks)
else()
    message(STATUS "Benchmarks not found. To include them, clone https://github.com/EnzymeAD/enzyme-benchmarks into the Enzyme source tree.")
endif()

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
set(var INSTALL_${p}_DIR)
if(NOT IS_ABSOLUTE "${${var}}")
  set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
endif()
endforeach()

if (ENZYME_ENABLE_PLUGINS)
export(TARGETS LLVMEnzyme-${LLVM_VERSION_MAJOR}
  FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")

if (${Clang_FOUND})
export(TARGETS ClangEnzyme-${LLVM_VERSION_MAJOR}
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
export(TARGETS ClangEnzymeFlags
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")

file(GLOB_RECURSE ENZYME_INCLUDES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
set(bundled_includes "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/Clang/bundled_includes.h")
list(APPEND ENZYME_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle-includes.sh)
add_custom_command(OUTPUT ${bundled_includes}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle-includes.sh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/enzyme/enzyme
  ${bundled_includes}
  DEPENDS ${ENZYME_INCLUDES}
  COMMENT "Bundling includes"
  VERBATIM
)
add_custom_target(bundled_includes_target DEPENDS ${bundled_includes})
add_dependencies(ClangEnzyme-${LLVM_VERSION_MAJOR} bundled_includes_target)
target_include_directories(ClangEnzyme-${LLVM_VERSION_MAJOR} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/Clang/")
endif()
endif()

if (${ENZYME_FLANG} AND ${Clang_FOUND})
	add_custom_target(link_target ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink
	  ClangEnzyme-${LLVM_VERSION_MAJOR}${LLVM_SHLIBEXT}
	  ${PROJECT_BINARY_DIR}/Enzyme/FlangEnzyme-${LLVM_VERSION_MAJOR}${LLVM_SHLIBEXT}
  )
endif()

if (ENZYME_ENABLE_PLUGINS)
export(TARGETS LLDEnzyme-${LLVM_VERSION_MAJOR}
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
endif()

export(TARGETS LLDEnzymeFlags
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
export(TARGETS LLDEnzymeAssumeUnknownNoFree
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
export(TARGETS LLDEnzymeLooseTypeFlags
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
export(TARGETS LLDEnzymePrintTypeFlags
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
export(TARGETS LLDEnzymePrintFlags
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")
export(TARGETS LLDEnzymeNoStrictAliasingFlags
    APPEND FILE "${PROJECT_BINARY_DIR}/EnzymeTargets.cmake")

export(PACKAGE Enzyme)

set(CONF_LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
set(CONF_LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR})
set(CONF_LLVM_VERSION_PATCH ${LLVM_VERSION_PATCH})
set(CONF_LLVM_DIR ${LLVM_DIR})
set(CONF_LLVM_BINARY_DIR ${LLVM_BINARY_DIR})
if (${Clang_FOUND})
    get_property(CONF_CLANG_EXE TARGET clang PROPERTY LOCATION)
else()
    set(CONF_CLANG_EXE "")
endif()

set(CONF_INCLUDE_DIRS lib)
configure_file(cmake/EnzymeConfig.cmake.in
     "${PROJECT_BINARY_DIR}/EnzymeConfig.cmake" @ONLY)

set(CONF_INCLUDE_DIRS lib)
configure_file(cmake/EnzymeConfig.cmake.in
     "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EnzymeConfig.cmake" @ONLY)

configure_file(cmake/EnzymeConfigVersion.cmake.in
     "${PROJECT_BINARY_DIR}/EnzymeConfigVersion.cmake" @ONLY)

install(FILES
     "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/EnzymeConfig.cmake"
     "${PROJECT_BINARY_DIR}/EnzymeConfigVersion.cmake"
     DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT EnzymeTargets
  DESTINATION "${INSTALL_CMAKE_DIR}"
  COMPONENT   dev)
