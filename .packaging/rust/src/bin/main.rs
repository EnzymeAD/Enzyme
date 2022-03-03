use enzyme::Repo::{self, *};

fn main() {
    let args = Cli::parse();
    enzyme::download(args);
    enzyme::build(args);
    if let Some(enzyme) = args.enzyme {
        enzyme::generate_bindings(enzyme)?;
    }
}
