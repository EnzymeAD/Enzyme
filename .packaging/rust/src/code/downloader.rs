use super::utils::{self, run_and_printerror};
use super::version_manager::*;
use crate::{Cli, Repo};

use curl::easy::Easy;
use flate2;
use flate2::bufread::GzDecoder;
use std::fs::{remove_dir_all, File, OpenOptions};
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::Command;
use tar::Archive;

/// Download the given tarball to the specified location
fn download_tarball(repo_url: &str, download_filename: PathBuf) -> Result<(), String> {
    if std::path::Path::new(&download_filename).exists() {
        match std::fs::remove_file(&download_filename) {
            Ok(()) => {}
            Err(e) => {
                return Err(format!(
                    "unable to delete file, download location blocked: {}",
                    e
                ))
            }
        }
    }

    let mut file = match OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .open(&download_filename)
    {
        Ok(file_handler) => file_handler,
        Err(why) => panic!("couldn't create {}", why),
    };

    dbg!(&repo_url);
    let mut handle = Easy::new();
    handle.url(repo_url).unwrap();
    handle.follow_location(true).unwrap();

    let mut data = Vec::new();
    {
        let mut transfer = handle.transfer();
        transfer
            .write_function(|block_data| {
                data.extend_from_slice(block_data);
                Ok(block_data.len())
            })
            .unwrap();
        transfer.perform().unwrap();
    }

    dbg!(&download_filename);
    match file.write_all(&data) {
        Ok(_) => {}
        Err(e) => {
            return Err(format!(
                "Unable to write download {} to file {}. {}",
                repo_url,
                download_filename.display(),
                e
            ))
        }
    };
    file.sync_all().unwrap();
    Ok(())
}

fn unpack(tar_gz_file: &str, dst: &str) -> Result<(), std::io::Error> {
    let path = tar_gz_file;

    let tar_gz = File::open(path)?;
    let tar_gz_reader = BufReader::new(tar_gz);
    let tar = GzDecoder::new(tar_gz_reader);
    let mut archive = Archive::new(tar);
    archive.unpack(dst)?;

    Ok(())
}

fn download_head(repo_url: &str, out_dir: PathBuf) -> Result<(), String> {
    //let out_dir = utils::get_enzyme_base_path().join("Enzyme-HEAD");
    if out_dir.exists() {
        // make space to download the latest head
        remove_dir_all(out_dir.clone()).expect("failed to delete existing directory!");
    }
    let mut command = Command::new("git");
    command.args(&["clone", "--depth", "1", repo_url, out_dir.to_str().unwrap()]);
    run_and_printerror(&mut command);
    Ok(())
}

/// This function can be used to download enzyme / rust from github.
///
/// Stable released are downloaded as tarballs and unpacked, the Head is taken from the github repo
/// directly. Data will be processed in `~/.cache/enzyme`.
/// Will not perform any action for those which are set to None or Some(Local(_)).
pub fn download(to_download: Cli) -> Result<(), String> {
    // If we have alrady downloaded it in the past, there's nothing left to do.
    // if check_downloaded(&to_download) {
    //     return Ok(());
    // }
    if let Some(rust) = to_download.rust {
        match rust {
            Repo::Local(_) => {}
            Repo::Stable => {
                let remote_tarball = utils::get_remote_rustc_tarball_path();
                let name = "rustc";
                // Location to store our tarball, before unpacking
                let download_filename = utils::get_download_dir().join(name.to_owned() + ".tar.gz");
                download_head(&remote_tarball, download_filename.clone())?;
                let dest_dir = utils::get_rustc_repo_path();
                match unpack(
                    download_filename.to_str().unwrap(),
                    dest_dir.to_str().unwrap(),
                ) {
                    Ok(_) => {}
                    Err(e) => return Err(format!("failed unpacking: {}", e)),
                };
            }
            Repo::Head => {
                let remote_path = "https://github.com/rust-lang/rust";
                let name = "rust";
                // Location to store our tarball, before unpacking
                // TODO: Fix, directly download into repo
                download_head(&remote_path, download_filename.clone())?;
            }
        };
        // Mark it as downloaded, so we can skip the download step next time.
        // TODO
        // set_downloaded(&to_download);
    }

    if let Some(enzyme) = to_download.enzyme {
        match enzyme {
            Repo::Local(_) => {}
            Repo::Stable => {
                let remote_tarball = utils::get_remote_enzyme_tarball_path();
                let name = "rustc";
                // Location to store our tarball, before unpacking
                let download_filename = utils::get_download_dir().join(name.to_owned() + ".tar.gz");
                download_tarball(&remote_tarball, download_filename.clone())?;
                let dest_dir = utils::get_enzyme_base_path();
                match unpack(
                    download_filename.to_str().unwrap(),
                    dest_dir.to_str().unwrap(),
                ) {
                    Ok(_) => {}
                    Err(e) => return Err(format!("failed unpacking: {}", e)),
                };
            }
            Repo::Head => {
                let remote_path = "https://github.com/EnzymeAD/Enzyme";
                let name = "enzyme";
                // Location to store our tarball, before unpacking
                let download_filename = utils::get_download_dir().join(name.to_owned() + ".tar.gz");
                download_head(&remote_path, download_filename.clone())?;
            }
        };
    }
    Ok(())
}
