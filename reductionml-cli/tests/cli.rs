use assert_cmd::prelude::*;
use assert_fs::prelude::FileWriteStr;
// Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs

#[test]
fn invalid_config_fails_check() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("reml")?;

    let file = assert_fs::NamedTempFile::new("config.json")?;
    file.write_str("{}")?;

    cmd.arg("config").arg("check").arg(file.path());
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to parse configuration"));
    Ok(())
}
