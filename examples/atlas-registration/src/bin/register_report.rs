// SPDX-License-Identifier: MIT
// Original work for this crate.

//! Print a registration measurement report as JSON, and fail if the native
//! transform is outside the tolerance.

use std::path::PathBuf;

use blockflow_atlas_registration::report::{
    fixture_report, render_json, require_threshold, synthetic_translation_report,
};
use clap::Parser;

/// Measure native registration against a known translation, or against a
/// recorded elastix run.
#[derive(Parser)]
#[command(name = "register-report")]
struct Cli {
    /// A fixture directory with `registration-manifest.json`; without one the
    /// synthetic translation is measured.
    #[arg(long)]
    fixture: Option<PathBuf>,
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("register-report: use a release build (cargo run --release)");
        std::process::exit(2);
    }
    let cli = Cli::parse();
    let report = match &cli.fixture {
        Some(root) => fixture_report(root),
        None => synthetic_translation_report(),
    };
    match report.and_then(|report| {
        require_threshold(&report)?;
        Ok(report)
    }) {
        Ok(report) => print!("{}", render_json(&report)),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
