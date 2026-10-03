# Resolves the Herbie executable the rewriting pass shells out to: a prebuilt
# one, or an ExternalProject build.
#
# No platform is compiled into the binary. Herbie's algebraic search is ranked
# by a per-device cost table, and the device is not known at build time, so the
# platform is generated from each device's cost model CSV
# (poseidon-calibrate --only herbie-platform) and passed to Herbie by path. The
# Herbie tree is not patched.
set(POSEIDON_HERBIE_BINARY ""
    CACHE
      FILEPATH
      "Prebuilt Herbie executable to use instead of building the herbie ExternalProject")
# The paper GPU's model; the CUDA lit tests price from it.
set(HERBIE_PLATFORM_CSV
    "${CMAKE_CURRENT_SOURCE_DIR}/cost_models/cm_sm_120_RTX5090.csv"
    CACHE
      FILEPATH "Cost model CSV the shipped Herbie platform was generated from")

if(POSEIDON_HERBIE_BINARY)
  set(POSEIDON_HERBIE_BINARY_PATH "${POSEIDON_HERBIE_BINARY}")
else()
  include(ExternalProject)
  externalproject_add(
    herbie
    GIT_REPOSITORY https://github.com/herbie-fp/herbie
    GIT_TAG 73ba1fe76f97d4cdb852a602addd82e3a983b209
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/herbie-prefix
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE
      1
      # Built from the source tree, not installed as a Racket package: the
      # module names raco exe embeds are then `syntax/platform-language`, which
      # is what the shipped cost_models/*.herbie.rkt platform files name.
    BUILD_COMMAND cargo build --release --manifest-path=egg-herbie/Cargo.toml
    COMMAND
      raco
      pkg
      install
      --auto
      --no-docs
      --batch
      ./egg-herbie
    COMMAND
      sh -c
      "raco pkg update --auto --no-docs --batch fpbench rival rival3 || raco pkg install --auto --no-docs --batch fpbench rival rival3"
    COMMAND mkdir -p herbie-compiled/
    COMMAND
      raco
      exe
      -o
      herbie
      --orig-exe
      --embed-dlls
      --vv
      src/main.rkt
    COMMAND raco distribute herbie-compiled herbie
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E make_directory
      ${CMAKE_CURRENT_BINARY_DIR}/herbie/install
    COMMAND
      ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_BINARY_DIR}/herbie-prefix/src/herbie/herbie-compiled
      ${CMAKE_CURRENT_BINARY_DIR}/herbie/install/herbie)
  set(POSEIDON_HERBIE_BINARY_PATH
      "${CMAKE_CURRENT_BINARY_DIR}/herbie/install/herbie/bin/herbie")
endif()
set(POSEIDON_HERBIE_BINARY_PATH "${POSEIDON_HERBIE_BINARY_PATH}"
    CACHE INTERNAL "Resolved Herbie executable path")
message(STATUS "Poseidon: herbie = ${POSEIDON_HERBIE_BINARY_PATH}")
