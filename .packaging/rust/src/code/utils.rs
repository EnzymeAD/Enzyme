use dirs;
use std::{path::PathBuf, process::Command, str::FromStr};

use super::version_manager::{ENZYME_VER, RUSTC_VER};

pub(crate) fn run_and_printerror(command: &mut Command) {
    println!("Running: `{:?}`", command);
    match command.status() {
        Ok(status) => {
            if !status.success() {
                panic!("Failed: `{:?}` ({})", command, status);
            }
        }
        Err(error) => {
            panic!("Failed: `{:?}` ({})", command, error);
        }
    }
}

/// We offer support for downloading and compiling these two repositories.
#[derive(Clone, Debug)]
pub enum Repo {
    /// For handling the Enzyme repository (latest release).
    Enzyme,
    /// For handling the Enzyme repository (main branch).
    EnzymeHEAD,
    /// For handling the Rust repository.
    Rust,
}
impl FromStr for Repo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower_s = s.to_lowercase();
        match lower_s.as_str() {
            "rust" => Ok(Repo::Rust),
            "enzyme" => Ok(Repo::Enzyme),
            "enzyme-head" => Ok(Repo::EnzymeHEAD),
            _ => Err(
                "The only supported parameters are \"rust\", \"enzyme\", or \"enzyme-head\""
                    .to_string(),
            ),
        }
    }
}

fn assert_existence(path: PathBuf) {
    if !path.is_dir() {
        std::fs::create_dir_all(path.clone())
            .unwrap_or_else(|_| panic!("Couldn't create: {}", path.display()));
    }
}
pub fn get_enzyme_base_path() -> PathBuf {
    let cache_dir = dirs::cache_dir().expect("Enzyme needs access to your cache dir.");
    dbg!(&cache_dir);
    let enzyme_base_path = cache_dir.join("enzyme");
    assert_existence(enzyme_base_path.clone());
    dbg!(&enzyme_base_path);
    enzyme_base_path
}
pub fn get_enzyme_repo_path() -> PathBuf {
    let path = get_enzyme_base_path().join("Enzyme-".to_owned() + ENZYME_VER);
    assert_existence(path.clone());
    path
}
fn get_enzyme_subdir_path() -> PathBuf {
    let path = get_enzyme_repo_path().join("enzyme");
    assert_existence(path.clone());
    path
}
pub fn get_capi_path() -> PathBuf {
    get_enzyme_subdir_path().join("Enzyme").join("CApi.h")
}
pub fn get_bindings_string() -> PathBuf {
    get_enzyme_base_path().join("enzyme.rs")
}
pub fn get_enzyme_build_path() -> PathBuf {
    let enzyme_path = get_enzyme_subdir_path().join("build");
    assert_existence(enzyme_path.clone());
    enzyme_path
}
pub fn get_download_dir() -> PathBuf {
    let enzyme_download_path = get_enzyme_base_path().join("downloads");
    assert_existence(enzyme_download_path.clone());
    enzyme_download_path
}
pub fn get_rustc_repo_path() -> PathBuf {
    let rustc_path = get_enzyme_base_path().join("rustc-".to_owned() + RUSTC_VER + "-src");
    assert_existence(rustc_path.clone());
    rustc_path
}
pub fn get_rustc_build_path() -> PathBuf {
    let rustc_path = get_rustc_repo_path().join("build");
    assert_existence(rustc_path.clone());
    rustc_path
}
fn get_rustc_platform_path() -> PathBuf {
    let platform = env!("TARGET");
    get_rustc_build_path().join(&platform)
}
pub fn get_rustc_stage2_path() -> PathBuf {
    get_rustc_platform_path().join("stage2")
}
pub fn get_llvm_build_path() -> PathBuf {
    get_rustc_platform_path().join("llvm").join("build")
}
pub fn get_llvm_header_path() -> PathBuf {
    get_rustc_platform_path().join("llvm").join("include")
}
pub fn get_remote_enzyme_tarball_path() -> String {
    format!(
        "https://github.com/EnzymeAD/Enzyme/archive/refs/tags/v{}.tar.gz",
        ENZYME_VER
    )
}
pub fn get_remote_rustc_tarball_path() -> String {
    format!(
        "https://static.rust-lang.org/dist/rustc-{}-src.tar.gz",
        RUSTC_VER
    )
}
