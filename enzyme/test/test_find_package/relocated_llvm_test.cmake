set(enzyme_config_in
    "${ENZYME_TEST_SOURCE_DIR}/../../cmake/EnzymeConfig.cmake.in")
set(enzyme_config_version_in
    "${ENZYME_TEST_SOURCE_DIR}/../../cmake/EnzymeConfigVersion.cmake.in")

file(REMOVE_RECURSE "${TEST_WORK_DIR}")

# Scenario 1: Cellar 22.1.1 was removed after upgrades to 22.1.2 and 22.1.10,
# and find_package(Enzyme) is called before project() to select clang.
set(brew_root "${TEST_WORK_DIR}/brew")
set(enzyme_cmake_dir "${brew_root}/Cellar/enzyme/0.0.100/lib/cmake/Enzyme")
file(MAKE_DIRECTORY "${enzyme_cmake_dir}")

foreach(version IN ITEMS "21.1.9" "22.1.2" "22.1.10")
  set(cellar_llvm "${brew_root}/Cellar/llvm/${version}")
  file(MAKE_DIRECTORY "${cellar_llvm}/bin" "${cellar_llvm}/lib/cmake/llvm")
  file(WRITE "${cellar_llvm}/lib/cmake/llvm/LLVMConfigVersion.cmake"
       "set(PACKAGE_VERSION \"${version}\")\n")
  file(WRITE "${cellar_llvm}/bin/clang-22" "")
  file(WRITE "${cellar_llvm}/bin/llvm-tblgen" "")
endforeach()
if(WIN32)
  file(COPY "${TEST_C_COMPILER}" DESTINATION
       "${brew_root}/Cellar/llvm/22.1.10/bin")
else()
  file(CREATE_LINK "${TEST_C_COMPILER}"
       "${brew_root}/Cellar/llvm/22.1.10/bin/clang" SYMBOLIC)
endif()

set(ENZYME_VERSION "0.0.100")
set(CONF_LLVM_VERSION_MAJOR "22")
set(CONF_LLVM_VERSION_MINOR "1")
set(CONF_LLVM_VERSION_PATCH "1")
set(CONF_LLVM_BINARY_DIR "${brew_root}/Cellar/llvm/22.1.1")
set(CONF_LLVM_DIR "${CONF_LLVM_BINARY_DIR}/lib/cmake/llvm")
set(CONF_CLANG_EXE "${CONF_LLVM_BINARY_DIR}/bin/clang-22")
set(Enzyme_CONFIG_TABLEGEN_EXE "${CONF_LLVM_BINARY_DIR}/bin/llvm-tblgen")

configure_file(
  "${enzyme_config_in}" "${enzyme_cmake_dir}/EnzymeConfig.cmake"
  @ONLY)
configure_file(
  "${enzyme_config_version_in}" "${enzyme_cmake_dir}/EnzymeConfigVersion.cmake"
  @ONLY)
file(WRITE "${enzyme_cmake_dir}/EnzymeTargets.cmake"
     "add_library(ClangEnzymeFlags INTERFACE IMPORTED)\n")

set(consumer_source_dir "${TEST_WORK_DIR}/consumer_source")
set(consumer_build_dir "${TEST_WORK_DIR}/consumer_build")
file(MAKE_DIRECTORY "${consumer_source_dir}")
file(
  WRITE "${consumer_source_dir}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.16)
find_package(Enzyme REQUIRED CONFIG PATHS \"${enzyme_cmake_dir}\" NO_DEFAULT_PATH)
set(expected_prefix \"${brew_root}/Cellar/llvm/22.1.10\")
if(NOT Enzyme_LLVM_BINARY_DIR STREQUAL expected_prefix)
  message(FATAL_ERROR \"Expected \${expected_prefix}, got \${Enzyme_LLVM_BINARY_DIR}\")
endif()
if(NOT Enzyme_LLVM_DIR STREQUAL \"\${expected_prefix}/lib/cmake/llvm\")
  message(FATAL_ERROR \"Unexpected Enzyme_LLVM_DIR: \${Enzyme_LLVM_DIR}\")
endif()
if(NOT Enzyme_LLVM_VERSION_PATCH STREQUAL \"10\")
  message(FATAL_ERROR \"Unexpected patch: \${Enzyme_LLVM_VERSION_PATCH}\")
endif()
if(NOT Enzyme_CLANG_EXE STREQUAL \"\${expected_prefix}/bin/clang-22\")
  message(FATAL_ERROR \"Unexpected clang: \${Enzyme_CLANG_EXE}\")
endif()
if(NOT Enzyme_TABLEGEN_EXE STREQUAL \"\${expected_prefix}/bin/llvm-tblgen\")
  message(FATAL_ERROR \"Unexpected tablegen: \${Enzyme_TABLEGEN_EXE}\")
endif()
if(NOT \"LLDEnzyme-22\" IN_LIST Enzyme_LIBRARIES)
  message(FATAL_ERROR \"Missing LLDEnzyme-22 in \${Enzyme_LIBRARIES}\")
endif()
if(NOT WIN32)
  set(CMAKE_C_COMPILER \"\${Enzyme_LLVM_BINARY_DIR}/bin/clang\")
  project(relocated_consumer LANGUAGES C)
else()
  project(relocated_consumer LANGUAGES NONE)
endif()
")

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -S "${consumer_source_dir}" -B "${consumer_build_dir}"
  RESULT_VARIABLE consumer_result
  OUTPUT_VARIABLE consumer_output
  ERROR_VARIABLE consumer_error)
if(NOT consumer_result EQUAL 0)
  message(
    FATAL_ERROR
      "Relocated Cellar test failed (${consumer_result}):\n${consumer_output}\n${consumer_error}")
endif()

# Scenario 2: Build-host Cellar directory does not exist at all; resolve via
# <prefix>/opt/llvm relative to Enzyme's own install location.
set(relocated_root "${TEST_WORK_DIR}/relocated_brew")
set(relocated_enzyme_cmake
    "${relocated_root}/Cellar/enzyme/0.0.100/lib/cmake/Enzyme")
file(
  MAKE_DIRECTORY "${relocated_enzyme_cmake}"
  "${relocated_root}/opt/llvm/lib/cmake/llvm" "${relocated_root}/opt/llvm/bin")
file(WRITE "${relocated_root}/opt/llvm/lib/cmake/llvm/LLVMConfigVersion.cmake"
     "set(PACKAGE_VERSION \"22.1.3\")\n")
file(WRITE "${relocated_root}/opt/llvm/bin/clang" "")

set(CONF_LLVM_BINARY_DIR "${TEST_WORK_DIR}/missing_host/Cellar/llvm/22.1.1")
set(CONF_LLVM_DIR "${CONF_LLVM_BINARY_DIR}/lib/cmake/llvm")
set(CONF_CLANG_EXE "${CONF_LLVM_BINARY_DIR}/bin/clang-22")
set(Enzyme_CONFIG_TABLEGEN_EXE "")

configure_file(
  "${enzyme_config_in}" "${relocated_enzyme_cmake}/EnzymeConfig.cmake"
  @ONLY)
configure_file(
  "${enzyme_config_version_in}"
  "${relocated_enzyme_cmake}/EnzymeConfigVersion.cmake"
  @ONLY)
file(WRITE "${relocated_enzyme_cmake}/EnzymeTargets.cmake"
     "add_library(ClangEnzymeFlags INTERFACE IMPORTED)\n")

set(opt_source_dir "${TEST_WORK_DIR}/opt_source")
set(opt_build_dir "${TEST_WORK_DIR}/opt_build")
file(MAKE_DIRECTORY "${opt_source_dir}")
file(
  WRITE "${opt_source_dir}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.16)
find_package(Enzyme REQUIRED CONFIG PATHS \"${relocated_enzyme_cmake}\" NO_DEFAULT_PATH)
set(expected_opt \"${relocated_root}/opt/llvm\")
if(NOT Enzyme_LLVM_BINARY_DIR STREQUAL expected_opt)
  message(FATAL_ERROR \"Expected \${expected_opt}, got \${Enzyme_LLVM_BINARY_DIR}\")
endif()
if(NOT Enzyme_LLVM_VERSION_PATCH STREQUAL \"3\")
  message(FATAL_ERROR \"Unexpected patch: \${Enzyme_LLVM_VERSION_PATCH}\")
endif()
if(NOT Enzyme_CLANG_EXE STREQUAL \"\${expected_opt}/bin/clang\")
  message(FATAL_ERROR \"Unexpected clang fallback: \${Enzyme_CLANG_EXE}\")
endif()
project(opt_consumer LANGUAGES NONE)
")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${opt_source_dir}" -B "${opt_build_dir}"
  RESULT_VARIABLE opt_result
  OUTPUT_VARIABLE opt_output
  ERROR_VARIABLE opt_error)
if(NOT opt_result EQUAL 0)
  message(
    FATAL_ERROR
      "Relocated opt/llvm test failed (${opt_result}):\n${opt_output}\n${opt_error}")
endif()
