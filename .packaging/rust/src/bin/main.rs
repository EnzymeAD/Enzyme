use enzyme::Repo::*;
fn main() {
    match get_enzyme() {
        Ok(()) => {}
        Err(e) => panic!("building failed: {}", e),
    }
}

fn get_enzyme() -> Result<(), String> {
    let mut cmd_line = std::env::args();
    if cmd_line.len() == 2 {
        let arg = cmd_line
            .nth(1)
            .expect("failed reading")
            .parse::<String>()
            .expect("failed reading");
        return match arg.to_lowercase().as_str() {
            "rust" => {
                enzyme::download(Rust)?;
                enzyme::build(Rust)?;
                Ok(())
            }
            "enzyme" => {
                enzyme::download(Enzyme)?;
                enzyme::generate_bindings_from_release()?;
                enzyme::build(Enzyme)?;
                Ok(())
            }
            "enzyme-head" => {
                enzyme::download(EnzymeHEAD)?;
                enzyme::generate_bindings_from_head()?;
                enzyme::build(EnzymeHEAD)?;
                Ok(())
            }
            _ => return Err(format!(
                "unknown input given {:?} \nPlease try \"rust\", \"enzyme\", or \"enzyme-head\"",
                &arg
            )),
        };
    } else if cmd_line.len() == 1 {
        // Default
        enzyme::download(Rust)?;
        enzyme::download(Enzyme)?;
        enzyme::build(Rust)?;
        enzyme::generate_bindings_from_release()?;
        enzyme::build(Enzyme)?;
        Ok(())
    } else {
        Err("To many arguments given".to_string())
    }
}
