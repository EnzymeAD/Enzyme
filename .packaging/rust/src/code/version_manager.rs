use std::fs::File;
use std::path::PathBuf;
use crate::Repo;
use super::utils;

pub const ENZYME_VER: &str = "0.0.25";
pub const RUSTC_VER: &str = "1.57.0";

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



pub fn check_downloaded(repo: &Repo) -> bool {
    match repo {
        Repo::Rust => rustc_download_finished().exists(),
        Repo::Enzyme => enzyme_download_finished().exists(),
    }
}

pub fn set_downloaded(repo: &Repo) {
    let path = match repo {
        Repo::Rust => rustc_download_finished(),
        Repo::Enzyme => enzyme_download_finished(),
    };
    File::create(&path).expect("Couldn't create downloaded-finished file");
}



pub fn check_compiled(repo: &Repo) -> bool {
    match repo {
        Repo::Rust => rustc_compile_finished().exists(),
        Repo::Enzyme => enzyme_compile_finished().exists(),
    }
}

pub fn set_compiled(repo: &Repo) {
    let path = match repo {
        Repo::Rust => rustc_compile_finished(),
        Repo::Enzyme => enzyme_compile_finished(),
    };
    File::create(&path).expect("Couldn't create compilation-finished file");
}
