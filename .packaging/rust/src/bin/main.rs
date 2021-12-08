use enzyme::Repo::*;
fn main() {
    match get_enzyme() {
        Ok(()) => {}
        Err(e) => panic!("building failed: {}", e),
    }
}

fn get_enzyme() -> Result<(), String> {
    enzyme::download(Rust)?;
    enzyme::download(Enzyme)?;
    enzyme::build(Rust)?;
    enzyme::generate_bindings()?;
    enzyme::build(Enzyme)?;
    Ok(())
}
