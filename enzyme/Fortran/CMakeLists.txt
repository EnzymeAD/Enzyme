enable_language(Fortran)
include(FortranCInterface)
FortranCInterface_VERIFY(QUIET)

add_library(EnzymeFortran enzyme.f90)
