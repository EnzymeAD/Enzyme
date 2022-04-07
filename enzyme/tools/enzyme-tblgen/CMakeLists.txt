include(TableGen)

set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(enzyme-tblgen Enzyme
  DerivativeGen.cpp
  enzyme-tblgen.cpp
  )

set_target_properties(enzyme-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(enzyme-tblgen
  PRIVATE
  EnzymeSupportIndentedOstream
  EnzymeTableGen)
set(Enzyme_CONFIG_TABLEGEN_EXE enzyme-tblgen)