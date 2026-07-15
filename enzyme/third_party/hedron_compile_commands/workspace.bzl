"""Loads Hedron's Compile Commands Extractor for Bazel."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

HEDRON_COMPILE_COMMANDS_COMMIT = "84c8aadfeee9a09105ec22cc85d0f478c90a788a"

def repo():
    http_archive(
        name = "hedron_compile_commands",
        integrity = "sha256-PIm60fdkep/HaUmYWZa3cmYcLj9PgYgnPTJRO8ooEtU=",
        strip_prefix = "bazel-compile-commands-extractor-" + HEDRON_COMPILE_COMMANDS_COMMIT,
        url = "https://github.com/vimarsh6739/bazel-compile-commands-extractor/archive/{commit}.tar.gz".format(
            commit = HEDRON_COMPILE_COMMANDS_COMMIT,
        ),
    )
