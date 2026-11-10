# A project that uses Poseidon through find_package(Poseidon) alone, in the
# mode POSEIDON_MODE names: a host function that is a site and, WITH_CUDA, the
# program of Inputs/two_tu as a static library of its two units and an
# application.
cmake_minimum_required(VERSION 3.18)
project(PoseidonPackage LANGUAGES CXX)

option(WITH_CUDA "Build the CUDA program of Inputs/two_tu as well" OFF)
set(POSEIDON_MODE plain CACHE STRING "plain, instrument or optimize")
set(POSEIDON_PROFILE "" CACHE PATH "Profile directory optimize mode reads")
set(POSEIDON_FLAGS "" CACHE STRING "Additional -poseidon-<flag>s")
set(POSEIDON_MIN_VERSION 0.1 CACHE STRING "Version find_package asks for")

find_package(Poseidon ${POSEIDON_MIN_VERSION} CONFIG REQUIRED)
if(DEFINED MPFR_FOUND)
  message(FATAL_ERROR "find_package(Poseidon) defined MPFR_FOUND")
endif()
get_target_property(plugin Poseidon::Poseidon-${POSEIDON_LLVM_VERSION_MAJOR}
                    LOCATION)
message(STATUS "Poseidon ${Poseidon_VERSION} from ${Poseidon_DIR}")
message(STATUS "Poseidon plugins ${plugin} ${POSEIDON_ENZYME_PLUGIN}")
message(STATUS "Poseidon include ${POSEIDON_INCLUDE_DIR}")

if(POSEIDON_MODE STREQUAL "plain")
  set(action Poseidon::Plain)
elseif(POSEIDON_MODE STREQUAL "instrument")
  set(action Poseidon::ProfileGenerate)
elseif(POSEIDON_MODE STREQUAL "optimize")
  set(action Poseidon::ProfileUse)
  set(POSEIDON_FLAGS "-poseidon-profile-use=${POSEIDON_PROFILE} ${POSEIDON_FLAGS}")
else()
  message(FATAL_ERROR "POSEIDON_MODE must be plain, instrument or optimize")
endif()
separate_arguments(POSEIDON_FLAGS UNIX_COMMAND "${POSEIDON_FLAGS}")
list(TRANSFORM POSEIDON_FLAGS PREPEND "SHELL:${POSEIDON_MLLVM} ")

add_executable(host host.cpp)
target_link_libraries(host PRIVATE ${action})
target_compile_options(host PRIVATE ${POSEIDON_FLAGS})

if(WITH_CUDA)
  enable_language(CUDA)
  add_library(kernels STATIC ${TWO_TU}/a.cu ${TWO_TU}/b.cu)
  target_include_directories(kernels PUBLIC ${TWO_TU})
  target_link_libraries(kernels PUBLIC ${action})
  target_compile_options(kernels PUBLIC ${POSEIDON_FLAGS})
  add_executable(app ${TWO_TU}/main.cu)
  target_link_libraries(app PRIVATE kernels)
endif()
