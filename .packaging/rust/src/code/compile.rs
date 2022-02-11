use super::{
    utils::{self, *},
    version_manager::{check_compiled, set_compiled},
};
use std::{path::PathBuf, process::Command};

#[allow(unused)]
fn print_llvm_version() {
    let mut cmake = Command::new("cmake");
    cmake.args(&["-version"]);
    run_and_printerror(&mut cmake);
}

/// A support function to build Enzyme and Rust such that they can be used together.
///
/// The Rust repository must be build first.
/// It will build a nightly version of Rust, together with LLVM and Clang.
/// Building the Enzyme repository will always run tests to verify that it is working correctly.
pub fn build(to_build: Repo) -> Result<(), String> {
    // If we have compiled that rustc or enzyme version in the past, we have nothing to do.
    // HEAD will always overwrite older HEAD versions, even if there was no change.
    if check_compiled(&to_build) {
        return Ok(());
    }

    let _repo = match to_build {
        Repo::Rust => build_rustc(),
        Repo::Enzyme => build_enzyme_release(),
        Repo::EnzymeHEAD => build_enzyme_head(),
    };

    // Compiling is expensive, so add a note that this compilation run was successfull and that
    // we shouldn't repeat it.
    set_compiled(&to_build);

    Ok(())
}

fn build_enzyme_head() {
    let enzyme_head_dir = utils::get_enzyme_base_path()
        .join("Enzyme-HEAD")
        .join("enzyme");
    assert!(
        enzyme_head_dir.exists(),
        "Apparently the previous cloning of HEAD failed?"
    );
    let build_dir = enzyme_head_dir.join("build");
    build_enzyme(build_dir);
}

fn build_enzyme_release() {
    let build_path = get_enzyme_build_path();
    build_enzyme(build_path);
}

fn build_enzyme(build_path: PathBuf) {
    let llvm_dir = get_llvm_build_path().join("lib").join("cmake").join("llvm");
    let llvm_dir = "-DLLVM_DIR=".to_owned() + llvm_dir.to_str().unwrap();
    let llvm_external_lit = get_rustc_repo_path()
        .join("src")
        .join("llvm-project")
        .join("llvm")
        .join("utils")
        .join("lit")
        .join("lit.py");
    let llvm_external_lit = "-DLLVM_EXTERNAL_LIT=".to_owned() + llvm_external_lit.to_str().unwrap();
    let llvm_external_lib = "-DENZYME_EXTERNAL_SHARED_LIB=ON".to_owned();
    let build_type = "-DCMAKE_BUILD_TYPE=Release";
    let mut cmake = Command::new("cmake");
    let mut ninja = Command::new("ninja");
    let mut ninja_check = Command::new("ninja");
    if !std::path::Path::new(&build_path).exists() {
        std::fs::create_dir(&build_path).unwrap();
    }
    cmake
        .args(&[
            "-G",
            "Ninja",
            "..",
            build_type,
            &llvm_external_lib,
            &llvm_dir,
            &llvm_external_lit,
        ])
        .current_dir(&build_path.to_str().unwrap());
    ninja.current_dir(&build_path.to_str().unwrap());
    ninja_check
        .args(&["check-enzyme"])
        .current_dir(&build_path.to_str().unwrap());
    run_and_printerror(&mut cmake);
    run_and_printerror(&mut ninja);
    run_and_printerror(&mut ninja_check);
}

fn build_rustc() {
    let mut cargo = Command::new("cargo");
    let mut configure = Command::new("./configure");
    let mut x = Command::new("x");
    let mut rustup = Command::new("rustup");

    let build_path = get_rustc_repo_path();

    if !std::path::Path::new(&build_path).exists() {
        std::fs::create_dir(&build_path).unwrap();
    }
    let x_path = std::path::Path::new("src").join("tools").join("x");

    let toolchain_path = get_rustc_stage2_path();

    cargo
        .args(&["install", "--path", x_path.to_str().unwrap()])
        .current_dir(&build_path.to_str().unwrap());

    configure
        .args(&[
            "--enable-llvm-link-shared",
            "--enable-llvm-plugins",
            "--release-channel=nightly",
            "--enable-llvm-assertions",
            "--enable-clang",
            "--enable-lld",
            "--enable-option-checking",
            "--enable-ninja",
        ])
        .current_dir(&build_path.to_str().unwrap());

    x.args(&["build", "--stage", "2"])
        .current_dir(&build_path.to_str().unwrap());

    rustup
        .args(&[
            "toolchain",
            "link",
            "enzyme",
            toolchain_path.to_str().unwrap(),
        ])
        .current_dir(&build_path.to_str().unwrap());

    run_and_printerror(&mut cargo);
    run_and_printerror(&mut configure);
    run_and_printerror(&mut x);
    run_and_printerror(&mut rustup);
}
