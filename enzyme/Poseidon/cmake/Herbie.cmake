# The Herbie executable the rewriting pass shells out to unless a compile names
# one with -poseidon-herbie-binary: a prebuilt one, one built from source, or
# none. No platform is compiled into the binary: the device is not known at
# build time, so each device's platform is generated from its cost model CSV
# and passed to Herbie by path.
set(POSEIDON_HERBIE_BINARY ""
    CACHE FILEPATH "Prebuilt Herbie executable the pass runs by default")
option(POSEIDON_BUILD_HERBIE
       "Build Herbie from source, which takes Rust and Racket" OFF)
if(POSEIDON_HERBIE_BINARY AND POSEIDON_BUILD_HERBIE)
  message(
    FATAL_ERROR
      "POSEIDON_HERBIE_BINARY and POSEIDON_BUILD_HERBIE each give the "
      "pass its Herbie; set one of them")
endif()
set(POSEIDON_HERBIE_BINARY_PATH "${POSEIDON_HERBIE_BINARY}")
if(POSEIDON_BUILD_HERBIE)
  include(ExternalProject)
  externalproject_add(
    herbie
    GIT_REPOSITORY https://github.com/herbie-fp/herbie
    GIT_TAG 73ba1fe76f97d4cdb852a602addd82e3a983b209
    PREFIX
      ${CMAKE_CURRENT_BINARY_DIR}/herbie-prefix
      # A pinned commit has nothing to update, and an update would rerun the
      # build step, whose raco pkg install refuses the package it installed.
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE
      1
      # Built from the source tree, not installed as a Racket package: the
      # module names raco exe embeds are then `syntax/platform-language`, the
      # language herbie_platform.py names in the platforms it writes.
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
    CACHE INTERNAL "Herbie executable the pass runs by default")
if(POSEIDON_HERBIE_BINARY_PATH)
  message(STATUS "Poseidon: herbie = ${POSEIDON_HERBIE_BINARY_PATH}")
else()
  message(STATUS "Poseidon: no default Herbie")
endif()
