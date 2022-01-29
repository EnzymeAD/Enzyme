use super::utils;
use bindgen;
use std::{fs, path::PathBuf};

/// This function can be used to generate Rust wrappers around Enzyme's [C API](https://github.com/wsmoses/Enzyme/blob/main/enzyme/Enzyme/CApi.h).
pub fn generate_bindings() -> Result<(), String> {
    let capi_header = utils::get_capi_path();
    let out_file = utils::get_bindings_string();
    generate_bindings_with(capi_header, out_file)
}

/// This function can be used to generate Rust wrappers around Enzyme's [C API](https://github.com/wsmoses/Enzyme/blob/main/enzyme/Enzyme/CApi.h).
pub fn generate_bindings_with(capi_header: PathBuf, out_file: PathBuf) -> Result<(), String> {
    dbg!(&capi_header);

    // tell cargo to re-run the builder if the header has changed
    println!("cargo:rerun-if-changed={}", capi_header.display());
    let content: String = fs::read_to_string(capi_header.clone()).unwrap();

    let bindings = bindgen::Builder::default()
        .header_contents("CApi.hpp", &content) // read it as .hpp so bindgen can ignore the class successfully
        .clang_args(&[format!("-I{}", utils::get_llvm_header_path().display())])
        //.blacklist_item("CustomFunctionForward")
        //.blacklist_item("DiffeGradientUtils")
        .allowlist_type("CConcreteType")
        .rustified_enum("CConcreteType")
        .allowlist_type("CDerivativeMode")
        .rustified_enum("CDerivativeMode")
        .allowlist_type("CDIFFE_TYPE")
        .rustified_enum("CDIFFE_TYPE")
        .allowlist_type("LLVMContextRef")
        .allowlist_type("CTypeTreeRef")
        .allowlist_type("EnzymeTypeAnalysisRef")
        .allowlist_function("EnzymeNewTypeTree")
        .allowlist_function("EnzymeNewTypeTreeCT")
        .allowlist_function("EnzymeFreeTypeTree")
        .allowlist_function("EnzymeMergeTypeTree")
        .allowlist_function("EnzymeTypeTreeOnlyEq")
        .allowlist_function("EnzymeMergeTypeTree")
        .allowlist_function("EnzymeTypeTreeShiftIndiciesEq")
        .allowlist_function("EnzymeTypeTreeToString")
        .allowlist_function("EnzymeTypeTreeToStringFree")
        // Next two are for debugging / printning type information
        .allowlist_function("EnzymeSetCLBool")
        .allowlist_function("EnzymeSetCLInteger")
        .allowlist_function("CreateTypeAnalysis")
        .allowlist_function("ClearTypeAnalysis")
        .allowlist_function("FreeTypeAnalysis")
        .allowlist_function("CreateEnzymeLogic")
        .allowlist_function("ClearEnzymeLogic")
        .allowlist_function("FreeEnzymeLogic")
        .allowlist_type("LLVMOpaqueModule")
        .allowlist_function("EnzymeCreatePrimalAndGradient")
        .allowlist_function("EnzymeCreateAugmentedPrimal")
        //.allowlist_function("LLVMModuleCreateWithName")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate();

    let bindings = match bindings {
        Ok(v) => v,
        Err(_) => {
            return Err(format!(
                "Unable to generate bindings from {}.",
                capi_header.display()
            ))
        }
    };

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    //let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()); // can't be used outside of build.rs
    if out_file.exists() {
        fs::remove_file(out_file.clone()).unwrap();
    }

    let result = bindings.write_to_file(out_file.clone());

    match result {
        Ok(_) => Ok(()),
        Err(_) => Err(format!(
            "Couldn't write bindings to {}.",
            out_file.display()
        )),
    }
}
