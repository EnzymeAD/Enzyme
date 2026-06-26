message("Building Fortran bindings")

enable_language(Fortran)
include(FortranCInterface)
FortranCInterface_VERIFY(QUIET)

# Initialize Fortran module build output directory for all subsequent targets
if(NOT DEFINED CMAKE_Fortran_MODULE_DIRECTORY)
  set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules")
endif()

add_library(EnzymeFortran SHARED enzyme.f90 enzyme_function_hooks.f90)

# Install library, create target file
install(
  TARGETS EnzymeFortran
  EXPORT EnzymeFortranTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/EnzymeFortran)

# Install Fortran module file
if(NOT DEFINED CMAKE_INSTALL_MODULEDIR)
  set(
    CMAKE_INSTALL_MODULEDIR
    "${CMAKE_INSTALL_INCLUDEDIR}/EnzymeFortran"
    CACHE STRING "Directory in prefix to install generated module files"
  )
endif()
install(FILES "${CMAKE_Fortran_MODULE_DIRECTORY}/enzyme.mod"
        DESTINATION "${CMAKE_INSTALL_MODULEDIR}")
