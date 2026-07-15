"""Loads Hedron's Compile Commands Extractor for Bazel."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
<<<<<<< HEAD
load("//:workspace.bzl", "HEDRON_COMPILE_COMMANDS_COMMIT", "HEDRON_COMPILE_COMMANDS_SHA256")


def repo():
    http_archive(
        name = "hedron_compile_commands",
        sha256 = HEDRON_COMPILE_COMMANDS_SHA256,
        strip_prefix = "bazel-compile-commands-extractor-" + HEDRON_COMPILE_COMMANDS_COMMIT,
        url = "https://github.com/vimarsh6739/bazel-compile-commands-extractor/archive/{commit}.tar.gz".format(
            commit = HEDRON_COMPILE_COMMANDS_COMMIT,
        ),
    )
