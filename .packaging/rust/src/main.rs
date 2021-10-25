fn main() {
    match get_enzyme() {
        Ok(()) => {},
        Err(e) => panic!("building failed: {}", e),
    }
}


fn get_enzyme() -> Result<(), String> {
    enzyme_build::download("rustc")?;
    enzyme_build::download("enzyme")?;
    enzyme_build::build("rustc")?;
    enzyme_build::generate_bindings()?;
    enzyme_build::build("enzyme")?;
    Ok(())
}
