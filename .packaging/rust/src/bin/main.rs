use enzyme::Repo::{self, *};
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
            .parse::<Repo>()
            .expect("failed reading first argument!");

        enzyme::download(arg.clone())?;
        enzyme::build(arg.clone())?;

        if let Repo::Enzyme = arg {
            enzyme::generate_bindings_from_release()?;
        } else if let Repo::EnzymeHEAD = arg {
            enzyme::generate_bindings_from_head()?;
        };

        Ok(())
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
