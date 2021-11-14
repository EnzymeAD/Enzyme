use dirs;
use std::fs;
use std::path::PathBuf;

const ENZYME_VER: &str = "0.0.20";
const RUSTC_VER: &str = "1.56.0";

pub fn clean_directory(path: PathBuf) {
    let mut path_reader = match path.read_dir() {
        Ok(r) => r,
        Err(e) => panic!("Can't read the directory: {}. {}", path.display(), e),
    };
    let is_empty = path_reader.next().is_none();
    if !is_empty {
        fs::remove_dir_all(path.clone()).unwrap();
        fs::create_dir(path).unwrap();
    }
}

fn assert_existence(path: PathBuf) {
    if !path.is_dir() {
        std::fs::create_dir_all(path.clone())
            .unwrap_or_else(|_| panic!("Couldn't create: {}", path.display()));
    }
}
pub fn get_enzyme_base_path() -> PathBuf {
    let cfg_dir = dirs::config_dir().expect("Enzyme needs access to your cfg dir.");
    dbg!(&cfg_dir);
    let enzyme_base_path = cfg_dir.join("enzyme");
    assert_existence(enzyme_base_path.clone());
    dbg!(&enzyme_base_path);
    enzyme_base_path
}
fn get_enzyme_subdir_path() -> PathBuf {
    let path = get_enzyme_base_path()
        .join("Enzyme-".to_owned() + ENZYME_VER)
        .join("enzyme");
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
    let platform = std::env::var("TARGET").unwrap();
    get_rustc_build_path().join(&platform)
}
pub fn get_rustc_stage2_path() -> PathBuf {
    get_rustc_platform_path().join("stage2")
}
pub fn get_rustc_binary_path() -> PathBuf {
    get_rustc_stage2_path().join("bin").join("rustc")
}
pub fn get_llvm_build_path() -> PathBuf {
    get_rustc_platform_path().join("llvm").join("build")
}
pub fn get_llvm_header_path() -> PathBuf {
    get_rustc_platform_path().join("llvm").join("include")
}
pub fn get_remote_enzyme_tarball_path() -> String {
    format!(
        "https://github.com/wsmoses/Enzyme/archive/refs/tags/v{}.tar.gz",
        ENZYME_VER
    )
}
pub fn get_remote_rustc_tarball_path() -> String {
    format!(
        "https://static.rust-lang.org/dist/rustc-{}-src.tar.gz",
        RUSTC_VER
    )
}
