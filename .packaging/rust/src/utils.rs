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
pub fn get_local_enzyme_base_path() -> PathBuf {
    let cfg_dir = dirs::config_dir().expect("Enzyme needs access to your cfg dir.");
    let enzyme_base_path = cfg_dir.join("enzyme");
    assert_existence(enzyme_base_path.clone());
    enzyme_base_path
}
fn get_local_enzyme_subdir_path() -> PathBuf {
    let path = get_local_enzyme_base_path()
        .join("Enzyme-".to_owned() + ENZYME_VER)
        .join("enzyme");
    assert_existence(path.clone());
    path
}

pub fn get_local_capi_path() -> PathBuf {
    
    get_local_enzyme_subdir_path().join("Enzyme").join("CApi.h")
}
pub fn get_local_bindings_string() -> String {
    let path = get_local_enzyme_base_path().join("enzyme.rs");
    path.to_str().unwrap().to_owned()
}
pub fn get_local_enzyme_build_path() -> PathBuf {
    let enzyme_path = get_local_enzyme_subdir_path().join("build");
    assert_existence(enzyme_path.clone());
    enzyme_path
}
pub fn get_local_download_dir() -> PathBuf {
    let enzyme_download_path = get_local_enzyme_base_path().join("downloads");
    assert_existence(enzyme_download_path.clone());
    enzyme_download_path
}
pub fn get_local_rustc_path() -> PathBuf {
    let rustc_path = get_local_enzyme_base_path().join("rustc-".to_owned() + RUSTC_VER + "-src");
    assert_existence(rustc_path.clone());
    rustc_path
}
pub fn get_local_rustc_build_path() -> PathBuf {
    let rustc_path = get_local_rustc_path().join("build");
    assert_existence(rustc_path.clone());
    rustc_path
}
pub fn get_local_llvm_build_path() -> PathBuf {
    let platform = std::env::var("TARGET").unwrap();
    
    get_local_rustc_build_path()
        .join(&platform)
        .join("llvm")
        .join("build")
}

pub fn get_remote_enzyme_tarball_path() -> PathBuf {
    let path = PathBuf::new();
    path.join(
        "https://github.com/wsmoses/Enzyme/archive/refs/tags/v".to_owned() + ENZYME_VER + ".tar.gz",
    )
}
pub fn get_remote_rustc_tarball_path() -> PathBuf {
    let path = PathBuf::new();
    path.join("https://static.rust-lang.org/dist/rustc-".to_owned() + RUSTC_VER + "-src.tar.gz")
}
