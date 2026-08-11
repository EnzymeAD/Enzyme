"""Bzlmod extensions for Hedron's Compile Commands Extractor."""

load(":workspace.bzl", hedron_compile_commands_repo = "repo")

def _hedron_compile_commands_impl(_):
    hedron_compile_commands_repo()

hedron_compile_commands = module_extension(
    implementation = _hedron_compile_commands_impl,
)
