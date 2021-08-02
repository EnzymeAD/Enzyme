fn main() {
    match get_enzyme() {
        Ok(()) => {},
        Err(e) => panic!("building failed: {}", e),
    }
}

/// Run a command and panic with error message if not succeeded
fn run_and_printerror(command: &mut std::process::Command) {
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

fn get_enzyme() -> Result<(), String> {
    //download("rustc")?;
    
    // We don't have everything in the release tarballs / stable yet, so use the gh repo for now
    let mut git = std::process::Command::new("git");
    git.args(&["clone", "https://github.com/rust-lang/rust", "--recursive", "rustc-1.54.0-src"]);
    git.current_dir(enzyme_build::utils::get_local_enzyme_base_path());
    run_and_printerror(&mut git);

    // Nightly upgraded to llvm-13 which isn't supported by enzyme yet.
    // Pinning the latest llvm-12 commit.
    let mut git = std::process::Command::new("git");
    git.args(&["checkout", "-b", "enzyme", "3cfb7305ddb7fd73b92c87ae6af1b169068b6b0f"]);
    git.current_dir(enzyme_build::utils::get_local_rustc_build_path());
    run_and_printerror(&mut git);

    enzyme_build::download("enzyme")?;
    enzyme_build::build("rustc")?;
    enzyme_build::generate_bindings()?;
    enzyme_build::build("enzyme")?;
    Ok(())
}
