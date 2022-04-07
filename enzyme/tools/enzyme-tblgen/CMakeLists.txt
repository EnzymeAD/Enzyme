set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
)

add_tablegen(enzyme-tblgen Enzyme
  AttrOrTypeDefGen.cpp
  AttrOrTypeFormatGen.cpp
  CodeGenHelpers.cpp
  DialectGen.cpp
  DirectiveCommonGen.cpp
  EnumsGen.cpp
  FormatGen.cpp
  LLVMIRConversionGen.cpp
  LLVMIRIntrinsicGen.cpp
  enzyme-tblgen.cpp
  OpClass.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  OpFormatGen.cpp
  OpGenHelpers.cpp
  OpInterfacesGen.cpp
  OpPythonBindingGen.cpp
  PassCAPIGen.cpp
  PassDocGen.cpp
  PassGen.cpp
  RewriterGen.cpp
  SPIRVUtilsGen.cpp
  StructsGen.cpp
  )

set_target_properties(enzyme-tblgen PROPERTIES FOLDER "Tablegenning")
target_link_libraries(enzyme-tblgen
  PRIVATE
  EnzymeSupportIndentedOstream
  EnzymeTableGen)

enzyme_check_all_link_libraries(enzyme-tblgen)
