use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Stream JSONL records from plain `.jsonl` files and, on native targets, `.jsonl.zst` files.
///
/// The callback returns `Ok(false)` to stop reading the current file early.
pub fn for_each_jsonl_line<F>(path: &str, mut f: F) -> Result<(), String>
where
    F: FnMut(&str) -> Result<bool, String>,
{
    if path.ends_with(".zst") {
        return for_each_zstd_line(path, f);
    }
    let file = fs::File::open(path).map_err(|e| format!("open input {path}: {e}"))?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read input line from {path}: {e}"))?;
        if !f(&line)? {
            break;
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn for_each_zstd_line<F>(path: &str, mut f: F) -> Result<(), String>
where
    F: FnMut(&str) -> Result<bool, String>,
{
    use std::process::{Command, Stdio};

    let mut child = Command::new("zstd")
        .args(["-dc", path])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn zstd -dc {path}: {e}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| format!("capture zstd stdout for {path}"))?;
    let reader = BufReader::new(stdout);
    let mut stopped_early = false;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read zstd line from {path}: {e}"))?;
        if !f(&line)? {
            stopped_early = true;
            break;
        }
    }
    if stopped_early {
        let _ = child.kill();
    }
    let status = child
        .wait()
        .map_err(|e| format!("wait for zstd -dc {path}: {e}"))?;
    if !stopped_early && !status.success() {
        return Err(format!("zstd -dc {path} failed with {status}"));
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn for_each_zstd_line<F>(_path: &str, _f: F) -> Result<(), String>
where
    F: FnMut(&str) -> Result<bool, String>,
{
    Err("zstd JSONL streaming is unavailable on wasm32 targets".to_string())
}

/// Compute a SHA-256 checksum for a completed cache artifact file.
#[cfg(not(target_arch = "wasm32"))]
pub fn sha256_file_hex(path: impl AsRef<Path>) -> Result<String, String> {
    use std::process::Command;

    let path = path.as_ref();
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .or_else(|_| {
            Command::new("shasum")
                .args(["-a", "256"])
                .arg(path)
                .output()
        })
        .map_err(|e| format!("spawn sha256 tool for {}: {e}", path.display()))?;
    if !output.status.success() {
        return Err(format!(
            "sha256 tool failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|e| format!("decode sha256 output for {}: {e}", path.display()))?;
    stdout
        .split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| format!("missing sha256 output for {}", path.display()))
}

#[cfg(target_arch = "wasm32")]
pub fn sha256_file_hex(_path: impl AsRef<Path>) -> Result<String, String> {
    Err("file SHA-256 checksums are unavailable on wasm32 targets".to_string())
}
