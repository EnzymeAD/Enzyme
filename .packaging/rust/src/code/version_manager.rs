use super::utils;
use crate::Repo;
use std::fs::File;
use std::path::PathBuf;

pub const ENZYME_VER: &str = "0.0.26";
pub const RUSTC_VER: &str = "1.58.1";

fn rustc_download_finished() -> PathBuf {
    utils::get_download_dir().join("rustc-".to_owned() + RUSTC_VER + "-ok")
}
fn enzyme_download_finished() -> PathBuf {
    utils::get_download_dir().join("enzyme-".to_owned() + ENZYME_VER + "-ok")
}
fn rustc_compile_finished() -> PathBuf {
    utils::get_rustc_build_path().join("rustc-".to_owned() + RUSTC_VER + "-ok")
}
fn enzyme_compile_finished() -> PathBuf {
    utils::get_enzyme_build_path().join("enzyme-".to_owned() + ENZYME_VER + "-ok")
}
fn enzyme_compile_head_finished() -> PathBuf {
    utils::get_enzyme_base_path()
        .join("Enzyme-HEAD")
        .join("enzyme")
        .join("build")
        .join("enzyme-HEAD-ok")
}

pub fn check_downloaded(repo: &Repo) -> bool {
    match repo {
        Repo::Rust => rustc_download_finished().exists(),
        Repo::Enzyme => enzyme_download_finished().exists(),
        Repo::EnzymeHEAD => false, // always trigger download of the latest head
    }
}

pub fn set_downloaded(repo: &Repo) {
    let path = match repo {
        Repo::Rust => rustc_download_finished(),
        Repo::Enzyme => enzyme_download_finished(),
        Repo::EnzymeHEAD => return, // we clone directly instead of downloading first
    };
    File::create(&path).expect("Couldn't create downloaded-finished file");
}

pub fn check_compiled(repo: &Repo) -> bool {
    match repo {
        Repo::Rust => rustc_compile_finished().exists(),
        Repo::Enzyme => enzyme_compile_finished().exists(),
        Repo::EnzymeHEAD => enzyme_compile_head_finished().exists(),
    }
}

pub fn set_compiled(repo: &Repo) {
    let path = match repo {
        Repo::Rust => rustc_compile_finished(),
        Repo::Enzyme => enzyme_compile_finished(),
        Repo::EnzymeHEAD => enzyme_compile_head_finished(),
    };
    File::create(&path).expect("Couldn't create compilation-finished file");
}
