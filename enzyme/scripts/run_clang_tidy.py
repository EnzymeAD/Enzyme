#!/usr/bin/env python3
"""Run clang-tidy across Enzyme C++ sources with two-tier check enforcement.

Why a dedicated runner instead of upstream `run-clang-tidy.py` / `clang-tidy-diff.py`:
1. Upstream wrapper scripts do not pass `--experimental-custom-checks`, which is
   required for `.clang-tidy` `CustomChecks` (`clang-query` AST matchers), and
   `clang-tidy-diff.py` forwards arguments after `--` as compiler flags rather
   than `clang-tidy` flags.
2. Enzyme enforces coding standards in two tiers:
   - Whole-tree checks (`WHOLE_TREE_CHECKS`): invariants enforced on every line
     of every Enzyme C++ translation unit and included Enzyme header (excluding
     `build/`, `SCEV/`, `llvm-project/`, `test/`).
   - Diff-scoped checks: all remaining checks configured in `.clang-tidy`
     (`enzyme/.clang-tidy`, `enzyme/Enzyme/MLIR/.clang-tidy`,
     `enzyme/Enzyme/MLIR/Passes/.clang-tidy`), enforced on added/modified lines
     when `--diff-base <rev>` is supplied, or across all lines with
     `--all-checks`.
3. On `--diff-base <rev>`, touched `.cpp` files (and `.cpp` files covering
   modified `.h` headers, or standalone `.h` headers via `-xc++-header`
   fallback) are analyzed in a single `clang-tidy` invocation with the full
   `.clang-tidy` check set and line-filtered in Python, while untouched `.cpp`
   files run only `WHOLE_TREE_CHECKS`. Diagnostics emitted from common headers
   are deduplicated across translation units.
"""

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

IMPL_SUFFIXES = (".cpp", ".cc")
HEADER_SUFFIXES = (".h", ".hpp")
CPP_SUFFIXES = IMPL_SUFFIXES + HEADER_SUFFIXES

EXCLUDED_DIRS: FrozenSet[str] = frozenset(
    {".git", "build", "enzyme-build", "SCEV", "llvm-project", "test", "third_party", "external"}
)
EXCLUDE_HEADER_REGEX = (
    r"(^|[/\\])(\.git|build|enzyme-build|bazel-[^/\\]*|llvm-project|SCEV|"
    r"third_party|external|include[/\\](llvm|llvm-c|clang|clang-c|mlir|mlir-c))"
    r"([/\\]|$)|\.inc$"
)

DEFAULT_SUBDIRS: Tuple[str, ...] = (
    "enzyme/Enzyme",
    "enzyme/tools",
)

WHOLE_TREE_CHECKS: Tuple[str, ...] = (
    "llvm-twine-local",
    "custom-llvm-twine-field-or-return",
    "portability-restrict-system-includes",
    "performance-avoid-endl",
    "custom-llvm-no-rtti-exceptions",
    "custom-llvm-no-iostream",
    "custom-llvm-std-sort",
    "custom-mlir-const-ir-pointer",
    "custom-mlir-builder-by-value",
    "custom-mlir-verifier-non-local",
    "custom-mlir-bool-error-return",
)
WHOLE_TREE_CHECK_SET: FrozenSet[str] = frozenset(WHOLE_TREE_CHECKS)

# Checks enabled only in `Enzyme/MLIR/.clang-tidy` (must not bleed into parent
# non-MLIR headers such as `Enzyme/Utils.h` when `#include`d from an MLIR TU).
MLIR_ONLY_CHECKS: FrozenSet[str] = frozenset(
    {
        "llvm-use-new-mlir-op-builder",
        "readability-redundant-inline-specifier",
        "readability-identifier-naming",
    }
)

DIAG_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<severity>warning|error|fatal error):\s+"
    r"(?P<message>.+?)\s+\[(?P<checks>[^\]]+)\]\s*$"
)
NOTE_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s+note:\s+(?P<message>.*)$"
)
INCLUDE_LINE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


@dataclass(frozen=True)
class Diagnostic:
    realpath: str
    display_path: str
    line: int
    col: int
    severity: str
    check: str
    message: str
    context_lines: Tuple[str, ...] = ()

    @property
    def is_compiler_error(self) -> bool:
        return self.check.startswith("clang-diagnostic-") and self.severity in (
            "error",
            "fatal error",
        )

    def dedup_key(self) -> Tuple[str, int, int, str, str]:
        return (self.realpath, self.line, self.col, self.check, self.message)


@dataclass(frozen=True)
class TaskSpec:
    path: str
    tu_dir: str
    full_checks: bool
    is_header: bool = False
    fallback_flags: Tuple[str, ...] = ()


@dataclass
class TaskResult:
    task: TaskSpec
    returncode: int
    diagnostics: List[Diagnostic]
    tool_error: Optional[str] = None


def _supports_custom_checks(binary: str) -> bool:
    try:
        proc = subprocess.run(
            [binary, "--experimental-custom-checks", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def find_clang_tidy(explicit: Optional[str] = None) -> str:
    """Locate the `clang-tidy` executable from CLI arg, `$CLANG_TIDY`, or PATH."""
    user_specified = bool(explicit or os.environ.get("CLANG_TIDY"))
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    elif os.environ.get("CLANG_TIDY"):
        candidates.append(os.environ["CLANG_TIDY"])
    else:
        candidates.extend(
            [
                "clang-tidy-24",
                "clang-tidy-23",
                "clang-tidy-22",
                "clang-tidy-21",
                "clang-tidy",
                "clang_tidy",
                "/usr/bin/clang-tidy",
                "/usr/bin/clang_tidy",
            ]
        )

    found: List[str] = []
    for cand in candidates:
        resolved = shutil.which(cand)
        if not resolved and os.path.isfile(cand) and os.access(cand, os.X_OK):
            resolved = os.path.abspath(cand)
        if resolved and resolved not in found:
            if user_specified or _supports_custom_checks(resolved):
                return resolved
            found.append(resolved)

    if found:
        return found[0]

    searched = ", ".join(candidates)
    raise FileNotFoundError(f"Unable to locate clang-tidy executable (tried: {searched})")


def find_git_toplevel(cwd: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return os.path.realpath(out) if out else None
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_default_roots(cwd: Optional[str] = None) -> List[str]:
    """Resolve default scan directories (`enzyme/Enzyme`, `enzyme/tools`)."""
    base = os.path.abspath(cwd or os.getcwd())
    cwd_roots = [os.path.join(base, rel) for rel in DEFAULT_SUBDIRS]
    if any(os.path.isdir(p) for p in cwd_roots):
        return [p for p in cwd_roots if os.path.isdir(p)]

    # Support running from inside `enzyme/` directly.
    short_subdirs = ("Enzyme", "tools")
    short_roots = [os.path.join(base, rel) for rel in short_subdirs]
    if all(os.path.isdir(p) for p in short_roots[:1]) and any(
        os.path.isdir(p) for p in short_roots[1:]
    ):
        return [p for p in short_roots if os.path.isdir(p)]

    # Fall back to locating `enzyme/` relative to `enzyme/scripts/run_clang_tidy.py`.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enzyme_dir = os.path.dirname(script_dir)
    script_roots = [os.path.join(enzyme_dir, rel) for rel in short_subdirs]
    if any(os.path.isdir(p) for p in script_roots):
        return [p for p in script_roots if os.path.isdir(p)]

    return cwd_roots


def _is_within(child: str, parent: str) -> bool:
    if child == parent:
        return True
    return child.startswith(parent.rstrip(os.sep) + os.sep)


def _has_excluded_segment(rel_path: str) -> bool:
    for part in rel_path.split(os.sep):
        if not part or part == ".":
            continue
        if part in EXCLUDED_DIRS or part.startswith("bazel-"):
            return True
    return False


def is_excluded_path(
    real_path: str,
    target_roots: Sequence[str],
    build_dir: Optional[str] = None,
    git_top: Optional[str] = None,
) -> bool:
    """Return True if `real_path` is inside an excluded subtree (`build/`, `SCEV/`, etc.)."""
    if build_dir and _is_within(real_path, build_dir):
        return True

    if git_top and _is_within(real_path, git_top):
        rel_git = os.path.relpath(real_path, git_top)
        # Check directory components relative to repository root (excluding filename).
        dir_rel = os.path.dirname(rel_git)
        if dir_rel and _has_excluded_segment(dir_rel):
            return True

    for root in target_roots:
        if _is_within(real_path, root):
            rel_root = os.path.relpath(real_path, root)
            dir_rel = os.path.dirname(rel_root)
            if dir_rel and _has_excluded_segment(dir_rel):
                return True
            return False

    return True


def is_target_cpp_file(
    real_path: str,
    target_roots: Sequence[str],
    build_dir: Optional[str] = None,
    git_top: Optional[str] = None,
) -> bool:
    if not real_path.endswith(CPP_SUFFIXES):
        return False
    return not is_excluded_path(
        real_path, target_roots=target_roots, build_dir=build_dir, git_top=git_top
    )


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: List[Tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_changed_intervals(
    diff_base: str, cwd: Optional[str] = None
) -> Dict[str, List[Tuple[int, int]]]:
    """Return `realpath -> [[start, end], ...]` of 1-based inclusive modified line ranges."""
    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=cwd, text=True
    ).strip()
    diff_out = subprocess.check_output(
        [
            "git",
            "-c",
            "diff.noprefix=false",
            "diff",
            "--no-ext-diff",
            "--unified=0",
            "--no-color",
            diff_base,
        ],
        cwd=top,
        text=True,
    )
    raw_intervals: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff_out.splitlines():
        if line.startswith("+++ b/"):
            rel = line[6:]
            current_file = os.path.realpath(os.path.join(top, rel))
        elif line.startswith("+++ /dev/null"):
            current_file = None
        elif current_file and line.startswith("@@ "):
            m = hunk_re.match(line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                if count > 0:
                    raw_intervals.setdefault(current_file, []).append(
                        (start, start + count - 1)
                    )
    return {path: _merge_intervals(ivs) for path, ivs in raw_intervals.items()}


def intervals_to_lines(
    intervals: Dict[str, List[Tuple[int, int]]]
) -> Dict[str, Set[int]]:
    changed: Dict[str, Set[int]] = {}
    for path, ivs in intervals.items():
        lines: Set[int] = set()
        for start, end in ivs:
            lines.update(range(start, end + 1))
        if lines:
            changed[path] = lines
    return changed


def compute_changed_lines(
    diff_base: str, cwd: Optional[str] = None
) -> Dict[str, Set[int]]:
    """Return mapping of realpath -> set of 1-based added/modified line numbers."""
    return intervals_to_lines(compute_changed_intervals(diff_base, cwd=cwd))


def load_compile_commands(build_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load `compile_commands.json` and deduplicate translation units by `realpath`."""
    compdb_path = os.path.join(build_dir, "compile_commands.json")
    if not os.path.isfile(compdb_path):
        raise FileNotFoundError(f"compile_commands.json not found in: {build_dir}")
    with open(compdb_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list):
        raise ValueError(f"Invalid compile_commands.json (expected JSON list): {compdb_path}")

    deduped: List[Tuple[str, Dict[str, Any]]] = []
    seen: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or "file" not in entry:
            continue
        tu_dir = entry.get("directory") or build_dir
        raw_file = entry["file"]
        if not os.path.isabs(raw_file):
            raw_file = os.path.join(tu_dir, raw_file)
        real_file = os.path.realpath(raw_file)
        if real_file in seen or not os.path.isfile(real_file):
            continue
        seen.add(real_file)
        deduped.append((real_file, entry))
    return deduped


def extract_compiler_flags(entry: Dict[str, Any], build_dir: str) -> List[str]:
    """Extract reusable compiler flags (`-I`, `-D`, `-std=...`, `-isystem`, etc.) from a compdb entry."""
    if "arguments" in entry and isinstance(entry["arguments"], list):
        tokens = [str(tok) for tok in entry["arguments"]]
    elif "command" in entry and isinstance(entry["command"], str):
        tokens = shlex.split(entry["command"])
    else:
        return []

    if not tokens:
        return []

    tu_dir = entry.get("directory") or build_dir
    raw_file = entry.get("file", "")
    abs_file = raw_file if os.path.isabs(raw_file) else os.path.join(tu_dir, raw_file)
    real_file = os.path.realpath(abs_file) if raw_file else ""

    flags: List[str] = []
    skip_next = False
    # Skip argv[0] (the compiler executable).
    for i, tok in enumerate(tokens[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if tok in ("-o", "-MF", "-MT", "-MQ", "-MJ"):
            skip_next = True
            continue
        if tok in ("-c", "-MD", "-MMD", "-MP"):
            continue
        if tok.startswith(("-o", "-MF", "-MT", "-MQ", "--output=")):
            continue
        # Skip the input source file argument.
        cand = tok if os.path.isabs(tok) else os.path.join(tu_dir, tok)
        if tok == raw_file or (real_file and os.path.realpath(cand) == real_file):
            continue
        # Convert relative `-Irel` / `-isystem rel` paths against `tu_dir` so they work anywhere.
        if tok in ("-I", "-isystem", "-iquote") and i + 1 < len(tokens):
            inc_path = tokens[i + 1]
            if not os.path.isabs(inc_path):
                inc_path = os.path.normpath(os.path.join(tu_dir, inc_path))
            flags.extend([tok, inc_path])
            skip_next = True
            continue
        if tok.startswith("-I") and len(tok) > 2:
            inc_path = tok[2:]
            if not os.path.isabs(inc_path):
                inc_path = os.path.normpath(os.path.join(tu_dir, inc_path))
            flags.append("-I" + inc_path)
            continue
        flags.append(tok)
    return flags


def _strip_comments_for_includes(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _extract_includes(cpp_path: str, cache: Dict[str, List[str]]) -> List[str]:
    if cpp_path in cache:
        return cache[cpp_path]
    try:
        with open(cpp_path, encoding="utf-8", errors="replace") as fh:
            cleaned = _strip_comments_for_includes(fh.read())
    except OSError:
        cache[cpp_path] = []
        return []
    incs = [
        m.group(1).replace("\\", "/")
        for line in cleaned.splitlines()
        for m in [INCLUDE_LINE_RE.match(line)]
        if m
    ]
    cache[cpp_path] = incs
    return incs


def _tu_includes_header(
    tu_realpath: str, header_realpath: str, include_cache: Dict[str, List[str]]
) -> bool:
    norm_hdr = header_realpath.replace("\\", "/")
    hdr_base = os.path.basename(header_realpath)
    tu_dir = os.path.dirname(tu_realpath)
    hdr_dir = os.path.dirname(header_realpath)
    for inc in _extract_includes(tu_realpath, include_cache):
        if inc.startswith("."):
            if os.path.realpath(os.path.join(tu_dir, inc)) == header_realpath:
                return True
        elif "/" in inc:
            if norm_hdr.endswith("/" + inc):
                return True
        elif inc == hdr_base:
            local_same_name = os.path.realpath(os.path.join(tu_dir, hdr_base))
            if os.path.isfile(local_same_name):
                if local_same_name == header_realpath:
                    return True
            elif _is_within(tu_dir, hdr_dir):
                return True
    return False


def _nearest_compdb_entry(
    target_path: str, all_entries: Sequence[Tuple[str, Dict[str, Any]]]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not all_entries:
        return None
    target_dir = os.path.dirname(target_path)

    def score(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, int, str]:
        tu_path, _ = item
        tu_dir = os.path.dirname(tu_path)
        try:
            common = os.path.commonpath([target_dir, tu_dir])
        except ValueError:
            common = ""
        # Prefer longest shared directory prefix, then shortest distance.
        common_depth = len(common.split(os.sep)) if common else 0
        rel_dist = len(os.path.relpath(tu_dir, common).split(os.sep)) if common else 999
        return (-common_depth, rel_dist, tu_path)

    return min(all_entries, key=score)


def iter_standalone_cpp_files(
    target_roots: Sequence[str],
    build_dir: Optional[str] = None,
    git_top: Optional[str] = None,
) -> List[str]:
    found: List[str] = []
    seen: Set[str] = set()
    for root in target_roots:
        if os.path.isfile(root):
            real = os.path.realpath(root)
            if real not in seen and is_target_cpp_file(real, target_roots, build_dir, git_top):
                seen.add(real)
                found.append(real)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in EXCLUDED_DIRS and not d.startswith("bazel-")
            )
            for name in sorted(filenames):
                if not name.endswith(IMPL_SUFFIXES):
                    continue
                real = os.path.realpath(os.path.join(dirpath, name))
                if real not in seen and is_target_cpp_file(
                    real, target_roots, build_dir, git_top
                ):
                    seen.add(real)
                    found.append(real)
    # If the caller passed header-only directories/files in standalone mode, include `.h`/`.hpp`.
    if not found:
        for root in target_roots:
            if os.path.isdir(root):
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = sorted(
                        d
                        for d in dirnames
                        if d not in EXCLUDED_DIRS and not d.startswith("bazel-")
                    )
                    for name in sorted(filenames):
                        if name.endswith(HEADER_SUFFIXES):
                            real = os.path.realpath(os.path.join(dirpath, name))
                            if real not in seen and is_target_cpp_file(
                                real, target_roots, build_dir, git_top
                            ):
                                seen.add(real)
                                found.append(real)
    return found


def plan_tasks(
    *,
    target_roots: Sequence[str],
    build_dir: Optional[str],
    compile_args: Optional[Sequence[str]],
    all_checks: bool,
    changed_lines: Optional[Dict[str, Set[int]]],
    git_top: Optional[str],
    cwd: str,
) -> List[TaskSpec]:
    """Build the minimal set of `clang-tidy` invocations covering whole-tree and diff checks."""
    if build_dir is None:
        files = iter_standalone_cpp_files(target_roots, build_dir=None, git_top=git_top)
        # In `--diff-base` standalone mode, also ensure modified headers under `target_roots`
        # not included by a scanned `.cpp` are checked directly.
        include_cache: Dict[str, List[str]] = {}
        full_set: Set[str] = set()
        if all_checks:
            full_set.update(files)
        elif changed_lines is not None:
            for f in files:
                if f in changed_lines:
                    full_set.add(f)

        extra_headers: List[str] = []
        if changed_lines is not None and not all_checks:
            for mod_path in sorted(changed_lines):
                if not mod_path.endswith(HEADER_SUFFIXES):
                    continue
                if not os.path.isfile(mod_path) or not is_target_cpp_file(
                    mod_path, target_roots, None, git_top
                ):
                    continue
                if any(_tu_includes_header(f, mod_path, include_cache) for f in full_set):
                    continue
                # Promote an existing `.cpp` if one includes this header.
                promoted = False
                for f in files:
                    if f.endswith(IMPL_SUFFIXES) and _tu_includes_header(
                        f, mod_path, include_cache
                    ):
                        full_set.add(f)
                        promoted = True
                        break
                if not promoted and mod_path not in files:
                    extra_headers.append(mod_path)
                    full_set.add(mod_path)

        tasks: List[TaskSpec] = []
        for f in [*files, *extra_headers]:
            tasks.append(
                TaskSpec(
                    path=f,
                    tu_dir=cwd,
                    full_checks=(f in full_set),
                    is_header=f.endswith(HEADER_SUFFIXES),
                    fallback_flags=tuple(compile_args or ()),
                )
            )
        return tasks

    all_compdb = load_compile_commands(build_dir)
    non_excluded_compdb = [
        (real_f, entry)
        for real_f, entry in all_compdb
        if real_f.endswith(IMPL_SUFFIXES)
        and not _is_within(real_f, build_dir)
        and not (
            git_top
            and _is_within(real_f, git_top)
            and _has_excluded_segment(os.path.dirname(os.path.relpath(real_f, git_top)))
        )
    ]

    # TUs explicitly inside `target_roots` (excluding `build/`, `SCEV/`, `llvm-project/`, `test/`).
    selected_entries: Dict[str, Dict[str, Any]] = {}
    full_check_tus: Set[str] = set()
    for real_f, entry in all_compdb:
        if is_target_cpp_file(real_f, target_roots, build_dir=build_dir, git_top=git_top):
            selected_entries[real_f] = entry
            if all_checks or (changed_lines is not None and real_f in changed_lines):
                full_check_tus.add(real_f)

    include_cache = {}
    header_targets: List[Tuple[str, bool]] = []

    # 1. Explicit `.h`/`.hpp` files passed in positional `paths`.
    for root in target_roots:
        if os.path.isfile(root) and root.endswith(HEADER_SUFFIXES):
            header_targets.append((root, all_checks or (changed_lines is not None and root in changed_lines)))

    # 2. Modified `.h`/`.hpp` headers in `--diff-base` mode.
    if changed_lines is not None and not all_checks:
        for mod_path in sorted(changed_lines):
            if mod_path.endswith(HEADER_SUFFIXES) and os.path.isfile(mod_path):
                if is_target_cpp_file(
                    mod_path, target_roots, build_dir=build_dir, git_top=git_top
                ):
                    header_targets.append((mod_path, True))

    standalone_header_tasks: Dict[str, TaskSpec] = {}
    for hdr_path, want_full in header_targets:
        already_covered = any(
            _tu_includes_header(tu, hdr_path, include_cache)
            for tu in (full_check_tus if want_full else selected_entries)
        )
        if already_covered:
            continue

        # Prefer promoting an existing `.cpp` from `compile_commands.json` that `#include`s `hdr_path`.
        candidates = [
            (real_f, entry)
            for real_f, entry in non_excluded_compdb
            if _tu_includes_header(real_f, hdr_path, include_cache)
        ]
        if candidates:
            best_tu, best_entry = _nearest_compdb_entry(hdr_path, candidates)  # type: ignore[misc]
            selected_entries.setdefault(best_tu, best_entry)
            if want_full:
                full_check_tus.add(best_tu)
            continue

        # Fallback for standalone / unincluded headers: run `clang-tidy` directly on the
        # `.h` file with `-extra-arg-before=-xc++-header` and borrowed compile flags.
        nearest = _nearest_compdb_entry(hdr_path, non_excluded_compdb or all_compdb)
        tu_dir = build_dir
        borrowed: Tuple[str, ...] = ()
        if nearest is not None:
            _, nearest_entry = nearest
            tu_dir = nearest_entry.get("directory") or build_dir
            borrowed = tuple(extract_compiler_flags(nearest_entry, build_dir))
        prev = standalone_header_tasks.get(hdr_path)
        standalone_header_tasks[hdr_path] = TaskSpec(
            path=hdr_path,
            tu_dir=tu_dir,
            full_checks=want_full or (prev.full_checks if prev else False),
            is_header=True,
            fallback_flags=borrowed,
        )

    tasks = [
        TaskSpec(
            path=real_f,
            tu_dir=entry.get("directory") or build_dir,
            full_checks=(real_f in full_check_tus),
            is_header=False,
        )
        for real_f, entry in selected_entries.items()
    ]
    tasks.extend(standalone_header_tasks.values())
    return tasks


def _resolve_diag_path(raw_path: str, tu_dir: str, cwd: str) -> Tuple[str, str]:
    if os.path.isabs(raw_path):
        real = os.path.realpath(raw_path)
    else:
        cand_tu = os.path.join(tu_dir, raw_path)
        if os.path.exists(cand_tu):
            real = os.path.realpath(cand_tu)
        else:
            real = os.path.realpath(os.path.join(cwd, raw_path))

    try:
        rel_cwd = os.path.relpath(real, cwd)
        display = rel_cwd if not rel_cwd.startswith("..") else real
    except ValueError:
        display = real
    return real, display


def parse_clang_tidy_output(stdout: str, tu_dir: str, cwd: str) -> List[Diagnostic]:
    """Parse `clang-tidy` stdout into structured `Diagnostic` objects with snippets."""
    diagnostics: List[Diagnostic] = []
    pending_header: Optional[Tuple[str, str, int, int, str, List[str], str]] = None
    pending_ctx: List[str] = []

    def flush() -> None:
        nonlocal pending_header, pending_ctx
        if pending_header is None:
            return
        real, display, line_no, col_no, sev, check_ids, msg = pending_header
        ctx = tuple(pending_ctx)
        for check_id in check_ids:
            diagnostics.append(
                Diagnostic(
                    realpath=real,
                    display_path=display,
                    line=line_no,
                    col=col_no,
                    severity=sev,
                    check=check_id,
                    message=msg,
                    context_lines=ctx,
                )
            )
        pending_header = None
        pending_ctx = []

    for raw_line in stdout.splitlines():
        m = DIAG_RE.match(raw_line)
        if m:
            flush()
            real, display = _resolve_diag_path(m.group("path"), tu_dir=tu_dir, cwd=cwd)
            raw_checks = [
                c.strip()
                for c in m.group("checks").split(",")
                if c.strip() and not c.strip().startswith("-")
            ]
            if not raw_checks:
                raw_checks = [m.group("checks").strip()]
            pending_header = (
                real,
                display,
                int(m.group("line")),
                int(m.group("col")),
                m.group("severity"),
                raw_checks,
                m.group("message"),
            )
            continue

        if NOTE_RE.match(raw_line):
            # Attach notes only up to the next top-level warning/error, or stop context capture.
            continue

        if pending_header is not None and (
            raw_line.startswith(" ") or raw_line.startswith("\t")
        ):
            if len(pending_ctx) < 4:
                pending_ctx.append(raw_line)
        elif raw_line.strip():
            flush()

    flush()
    return diagnostics


def run_single_task(
    task: TaskSpec,
    *,
    clang_tidy_bin: str,
    build_dir: Optional[str],
    compile_args: Optional[Sequence[str]],
    config_file: Optional[str],
    cwd: str,
) -> TaskResult:
    base_cmd = [
        clang_tidy_bin,
        "--experimental-custom-checks",
        "--quiet",
        "--warnings-as-errors=-*",
        "--header-filter=.*",
        f"--exclude-header-filter={EXCLUDE_HEADER_REGEX}",
    ]
    if config_file:
        base_cmd.append(f"--config-file={config_file}")
    if not task.full_checks:
        base_cmd.append("-checks=-*," + ",".join(WHOLE_TREE_CHECKS))
    if task.is_header:
        base_cmd.append("-extra-arg-before=-xc++-header")

    if build_dir is not None:
        cmd = [*base_cmd, "-p", build_dir, task.path]
    else:
        cmd = [*base_cmd, task.path, "--", *(compile_args or ())]

    proc = subprocess.run(cmd, text=True, capture_output=True)

    # If a standalone `.h` was not interpolated by `compile_commands.json`, retry with borrowed flags.
    if (
        build_dir is not None
        and task.is_header
        and "Compile command not found" in (proc.stderr or "")
    ):
        retry_cmd = [*base_cmd, task.path, "--", *task.fallback_flags]
        proc = subprocess.run(retry_cmd, text=True, capture_output=True)

    raw_diags = parse_clang_tidy_output(proc.stdout, tu_dir=task.tu_dir, cwd=cwd)
    task_norm = task.path.replace("\\", "/")
    in_mlir_tu = "/MLIR/" in task_norm
    in_passes_tu = "/MLIR/Passes/" in task_norm
    diags: List[Diagnostic] = []
    for d in raw_diags:
        diag_norm = d.realpath.replace("\\", "/")
        if (
            in_mlir_tu
            and "/MLIR/" not in diag_norm
            and (d.check in MLIR_ONLY_CHECKS or d.check.startswith("custom-mlir-"))
        ):
            continue
        if (
            in_passes_tu
            and "/MLIR/Passes/" not in diag_norm
            and d.check == "custom-mlir-pass-cl-opt"
        ):
            continue
        diags.append(d)
    tool_error: Optional[str] = None
    if proc.returncode != 0 and not any(d.is_compiler_error for d in diags):
        detail = (proc.stderr or proc.stdout or "").strip()
        tool_error = (
            f"clang-tidy exited with code {proc.returncode} on {task.path}"
            + (f":\n{detail}" if detail else "")
        )

    return TaskResult(
        task=task,
        returncode=proc.returncode,
        diagnostics=diags,
        tool_error=tool_error,
    )


def filter_diagnostics(
    diagnostics: Iterable[Diagnostic],
    *,
    target_roots: Sequence[str],
    build_dir: Optional[str],
    git_top: Optional[str],
    all_checks: bool = False,
    changed_lines: Optional[Dict[str, Set[int]]] = None,
    selected: Optional[Set[str]] = None,
    ignored: Optional[Set[str]] = None,
) -> List[Diagnostic]:
    """Filter diagnostics by two-tier rules and deduplicate across translation units."""
    kept_by_key: Dict[Tuple[str, int, int, str, str], Diagnostic] = {}
    for d in diagnostics:
        if d.is_compiler_error:
            kept_by_key.setdefault(d.dedup_key(), d)
            continue

        if not is_target_cpp_file(
            d.realpath, target_roots=target_roots, build_dir=build_dir, git_top=git_top
        ):
            continue
        if selected is not None and d.check not in selected:
            continue
        if ignored is not None and d.check in ignored:
            continue

        if all_checks or d.check in WHOLE_TREE_CHECK_SET:
            kept_by_key.setdefault(d.dedup_key(), d)
        elif changed_lines is not None:
            if d.line in changed_lines.get(d.realpath, set()):
                kept_by_key.setdefault(d.dedup_key(), d)

    return [kept_by_key[k] for k in sorted(kept_by_key)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="optional directories or files to scan (default: enzyme/Enzyme, enzyme/tools)",
    )
    parser.add_argument(
        "-p",
        "--build-dir",
        help="path to build directory containing compile_commands.json",
    )
    parser.add_argument(
        "--compile-args",
        action="append",
        help="compiler flags for standalone mode when -p/--build-dir is not used",
    )
    parser.add_argument(
        "--clang-tidy-binary",
        help="path to clang-tidy executable (default: $CLANG_TIDY or auto-detect)",
    )
    parser.add_argument(
        "--config-file",
        help="optional explicit .clang-tidy configuration file path",
    )
    parser.add_argument(
        "--diff-base",
        help="git revision against which diff-scoped checks are enforced on added/modified lines",
    )
    parser.add_argument(
        "--all-checks",
        action="store_true",
        help="enforce all .clang-tidy checks across all lines of scanned files",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="number of parallel clang-tidy worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--select",
        help="comma-separated subset of check IDs to report",
    )
    parser.add_argument(
        "--ignore",
        help="comma-separated check IDs to suppress",
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="emit GitHub Actions ::error annotations on stdout",
    )
    args = parser.parse_args(argv)

    if args.build_dir is None and args.compile_args is None:
        parser.error("one of -p/--build-dir or --compile-args is required")

    try:
        clang_tidy_bin = find_clang_tidy(args.clang_tidy_binary)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cwd = os.getcwd()
    build_dir: Optional[str] = None
    if args.build_dir is not None:
        raw_bd = os.path.abspath(args.build_dir)
        if os.path.isfile(raw_bd) and os.path.basename(raw_bd) == "compile_commands.json":
            raw_bd = os.path.dirname(raw_bd)
        build_dir = os.path.realpath(raw_bd)
        if not os.path.isfile(os.path.join(build_dir, "compile_commands.json")):
            print(
                f"error: compile_commands.json not found in build directory: {args.build_dir}",
                file=sys.stderr,
            )
            return 1

    compile_args: Optional[List[str]] = None
    if args.compile_args is not None:
        compile_args = []
        for item in args.compile_args:
            compile_args.extend(shlex.split(item))

    raw_roots = list(args.paths) if args.paths else resolve_default_roots(cwd=cwd)
    target_roots = [os.path.realpath(p) for p in raw_roots if os.path.exists(p)]
    if not target_roots:
        print(
            f"error: none of the target paths exist: {', '.join(raw_roots)}",
            file=sys.stderr,
        )
        return 1

    first_anchor = (
        target_roots[0]
        if os.path.isdir(target_roots[0])
        else os.path.dirname(target_roots[0])
    )
    git_top = find_git_toplevel(first_anchor)

    changed_lines: Optional[Dict[str, Set[int]]] = None
    if args.diff_base and not args.all_checks:
        try:
            changed_lines = compute_changed_lines(args.diff_base, cwd=first_anchor)
        except subprocess.CalledProcessError as exc:
            print(
                f"error: failed to compute git diff against {args.diff_base!r}: {exc}",
                file=sys.stderr,
            )
            return 1

    try:
        tasks = plan_tasks(
            target_roots=target_roots,
            build_dir=build_dir,
            compile_args=compile_args,
            all_checks=args.all_checks,
            changed_lines=changed_lines,
            git_top=git_top,
            cwd=cwd,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not tasks:
        print("checked 0 C++ translation units: no clang-tidy violations")
        return 0

    jobs = max(1, args.jobs)
    config_file = os.path.abspath(args.config_file) if args.config_file else None
    raw_diagnostics: List[Diagnostic] = []
    tool_errors: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                run_single_task,
                task,
                clang_tidy_bin=clang_tidy_bin,
                build_dir=build_dir,
                compile_args=compile_args,
                config_file=config_file,
                cwd=cwd,
            )
            for task in tasks
        ]
        for fut in futures:
            res = fut.result()
            raw_diagnostics.extend(res.diagnostics)
            if res.tool_error:
                tool_errors.append(res.tool_error)

    selected = (
        {s.strip() for s in args.select.split(",") if s.strip()}
        if args.select
        else None
    )
    ignored = (
        {s.strip() for s in args.ignore.split(",") if s.strip()}
        if args.ignore
        else None
    )

    findings = filter_diagnostics(
        raw_diagnostics,
        target_roots=target_roots,
        build_dir=build_dir,
        git_top=git_top,
        all_checks=args.all_checks,
        changed_lines=changed_lines,
        selected=selected,
        ignored=ignored,
    )

    for f in findings:
        print(f"{f.display_path}:{f.line}:{f.col}: error: [{f.check}] {f.message}")
        for ctx_line in f.context_lines:
            print(ctx_line)
        if args.github:
            print(
                f"::error file={f.display_path},line={f.line},"
                f"title=clang-tidy ({f.check})::{f.message}"
            )

    for err in tool_errors:
        print(f"error: {err}", file=sys.stderr)
        if args.github:
            first_line = err.splitlines()[0]
            print(f"::error title=clang-tidy execution error::{first_line}")

    if findings or tool_errors:
        if findings:
            print(
                f"\n{len(findings)} clang-tidy violation(s) found across {len(tasks)} translation unit(s). "
                f"Fix the violation or add `// NOLINT(<check>)` for an intentional exception.",
                file=sys.stderr,
            )
        return 1

    print(f"checked {len(tasks)} C++ translation unit(s): no clang-tidy violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
