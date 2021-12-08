use enzyme_build::Repo::*;
fn main() {
    match get_enzyme() {
        Ok(()) => {}
        Err(e) => panic!("building failed: {}", e),
    }
}

fn get_enzyme() -> Result<(), String> {
    enzyme_build::download(Rust)?;
    enzyme_build::download(Enzyme)?;
    enzyme_build::build(Rust)?;
    enzyme_build::generate_bindings()?;
    enzyme_build::build(Enzyme)?;
    Ok(())
}
