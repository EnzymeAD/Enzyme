use crate::utils;
use flate2;
use flate2::bufread::GzDecoder;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;
use tar::Archive;
use curl::easy::Easy;

fn unpack(tar_gz_file: &str, dst: &str) -> Result<(), std::io::Error> {
    let path = tar_gz_file;

    let tar_gz = File::open(path)?;
    let tar_gz_reader = BufReader::new(tar_gz);
    let tar = GzDecoder::new(tar_gz_reader);
    let mut archive = Archive::new(tar);
    archive.unpack(dst)?;

    Ok(())
}

pub fn download_tarball(repo_url: &str, download_filename: PathBuf) -> Result<(), String> {
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
        transfer.write_function(|block_data| {
            data.extend_from_slice(block_data);
            Ok(block_data.len())
        }).unwrap();
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

pub fn download(to_download: &str) -> Result<(), String> {
    let repo = match to_download {
        "enzyme" => utils::get_remote_enzyme_tarball_path(),
        "rustc" => utils::get_remote_rustc_tarball_path(),
        _ => {
            return Err(
                "Unknown download argument. Try enzyme xor rustc (includes llvm and others)."
                    .to_owned(),
            )
        }
    };

    // Caching fails if we have a new rustc / enzyme version
    // We would need to peek check if the new version would differ from the old one.
    // Thus deactivating caching for now
    let download_filename = utils::get_download_dir().join(to_download.to_owned() + ".tar.gz");
    download_tarball(&repo, download_filename.clone())?;
    /*
    if !Path::new(&download_filename).exists() { // We didn't cache this repo yet
        download_repo(repo, download_filename.clone())?;
    }
    */

    let dest_dir = utils::get_enzyme_base_path();

    dbg!(&dest_dir, &download_filename);
    match unpack(
        download_filename.to_str().unwrap(),
        dest_dir.to_str().unwrap(),
    ) {
        Ok(_) => {}
        Err(e) => return Err(format!("failed unpacking: {}", e)),
    };

    Ok(())
}
