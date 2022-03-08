#![allow(dead_code)]
use super::utils;
use crate::Cli;
use std::fs::File;

pub const ENZYME_VER: &str = "0.0.26";
pub const RUSTC_VER: &str = "1.58.1";

pub fn is_compiled_rust(args: &Cli) -> bool {
    utils::get_local_rust_repo_path(args.rust.clone())
        .join("compiled.txt")
        .is_file()
}
pub fn is_compiled_enzyme(args: &Cli) -> bool {
    utils::get_local_enzyme_repo_path(args.enzyme.clone())
        .join("compiled.txt")
        .is_file()
}
pub fn set_compiled_rust(args: &Cli) -> Result<(), String> {
    let repo = utils::get_local_rust_repo_path(args.rust.clone()).join("compiled.txt");
    match File::create(repo) {
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
pub fn set_compiled_enzyme(args: &Cli) -> Result<(), String> {
    let repo = utils::get_local_enzyme_repo_path(args.enzyme.clone()).join("compiled.txt");
    match File::create(repo) {
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}
