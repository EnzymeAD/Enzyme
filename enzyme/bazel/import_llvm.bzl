load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def _llvm_import_impl(ctx):
    new_local_repository(
        name = "llvm-raw",
        build_file_content = "# empty",
        # This path is relative to workspace root
        path = "../llvm-project",
    )

llvm_import_extension = module_extension(
    implementation = _llvm_import_impl,
)
