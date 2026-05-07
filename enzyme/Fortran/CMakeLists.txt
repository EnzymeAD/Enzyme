include(FortranCInterface)
FortranCInterface_VERIFY(QUIET)

add_library(EnzymeFortran enzyme_mod.f90)

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
install(FILES "${CMAKE_Fortran_MODULE_DIRECTORY}/enzyme_mod.mod"
        DESTINATION "${CMAKE_INSTALL_MODULEDIR}")
