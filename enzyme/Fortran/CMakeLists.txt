enable_language(Fortran)
include(FortranCInterface)
FortranCInterface_VERIFY(QUIET)

add_library(EnzymeFortran enzyme.f90)

# Install library, create target file
install(
  TARGETS EnzymeFortran
  EXPORT EnzymeFortranTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/EnzymeFortran)
