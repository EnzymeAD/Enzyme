# Locates the MPFR library the accuracy evaluator links against.
find_path(MPFR_INCLUDE_DIR mpfr.h)
find_library(MPFR_LIBRARY mpfr)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR DEFAULT_MSG MPFR_LIBRARY MPFR_INCLUDE_DIR)

if(MPFR_FOUND AND NOT TARGET MPFR::MPFR)
  add_library(MPFR::MPFR UNKNOWN IMPORTED)
  set_target_properties(MPFR::MPFR PROPERTIES
      IMPORTED_LOCATION "${MPFR_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${MPFR_INCLUDE_DIR}")
endif()
mark_as_advanced(MPFR_INCLUDE_DIR MPFR_LIBRARY)
