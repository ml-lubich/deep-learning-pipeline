//! End-to-end CLI tests (`dlpipe` binary).

use assert_cmd::Command;

#[test]
fn help_lists_commands_and_ascii_motif() {
    let assert = Command::cargo_bin("dlpipe")
        .expect("cargo built `dlpipe` binary")
        .arg("--help")
        .assert()
        .success();

    let hay = std::str::from_utf8(&assert.get_output().stdout).expect("utf8");
    assert!(hay.contains("D E E P   L E A R N I N G"));
    assert!(hay.contains("demo"));
    assert!(hay.contains("fit"));
}

#[test]
fn demo_runs_deterministic_report() {
    let assert = Command::cargo_bin("dlpipe")
        .expect("cargo built `dlpipe` binary")
        .arg("demo")
        .assert()
        .success();

    let hay = std::str::from_utf8(&assert.get_output().stdout).expect("utf8");
    assert!(hay.contains("deep learning pipeline sample"));
    assert!(hay.contains("train_rows"));
    assert!(hay.contains("val_mse"));
}

#[test]
fn fit_prints_mse_line() {
    let assert = Command::cargo_bin("dlpipe")
        .expect("cargo built `dlpipe` binary")
        .args(["fit", "--rows", "24", "--steps", "120", "--lr", "0.06"])
        .assert()
        .success();

    let hay = std::str::from_utf8(&assert.get_output().stdout).expect("utf8");
    assert!(hay.contains("train_mse="));
    assert!(hay.contains("val_mse="));
}

#[test]
fn demo_verbose_flag_still_succeeds() {
    Command::cargo_bin("dlpipe")
        .expect("cargo built `dlpipe` binary")
        .args(["--verbose", "demo"])
        .assert()
        .success();
}
