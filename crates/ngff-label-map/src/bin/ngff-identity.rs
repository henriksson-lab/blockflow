use std::path::Path;

use ngff_label_map::{migrate_legacy_identity_graph, validate_identity_group};

fn usage() -> ! {
    eprintln!("usage: ngff-identity validate LABEL_GROUP | migrate LABEL_GROUP LAYER_NAME");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| usage());
    let group = args.next().unwrap_or_else(|| usage());
    let result = match command.as_str() {
        "validate" if args.next().is_none() => validate_identity_group(Path::new(&group)),
        "migrate" => {
            let layer = args.next().unwrap_or_else(|| usage());
            if args.next().is_some() {
                usage();
            }
            migrate_legacy_identity_graph(Path::new(&group), &layer)
        }
        _ => usage(),
    };
    match result {
        Ok(metadata) => println!("{}", serde_json::to_string_pretty(&metadata).unwrap()),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
