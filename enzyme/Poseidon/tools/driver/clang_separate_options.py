#!/usr/bin/env python3
"""Write or check ClangSeparateOptions.def, the clang driver options whose value
can be the next argv entry, which poseidon-clang keeps paired with their option.

Usage:
    # From the Options.td of every clang Poseidon builds against, one
    # llvm-project checkout (or `git archive <tag> clang/include
    # llvm/include/llvm/Option cmake/Modules`) each, with one llvm-tblgen:
    python3 clang_separate_options.py generate <llvm-tblgen> <llvm-project>... \\
        > ClangSeparateOptions.def

    # Against an installed clang, without its sources (the lit suite does
    # this): every option of the file that the clang knows takes a separate
    # value there, and every option the clang offers for completion that takes
    # one is in the file or handled by poseidon-clang itself.
    python3 clang_separate_options.py check <clang> ClangSeparateOptions.def

The options are the Separate and JoinedOrSeparate ones visible to the clang
driver, of any of the given clangs. One that some of them do not take with a
separate value is listed with the LLVM versions that do. The MultiArg
and JoinedAndSeparate options are few and listed in poseidon-clang.cpp.
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile

HEADER = """\
//===- ClangSeparateOptions.def - clang options with a separate value -----===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The options of clang's driver whose value can be the next argv entry: the
// Separate and JoinedOrSeparate options that the clang driver accepts in
// Options.td of LLVM {versions}. CLANG_SEPARATE_OPTION_IN(name, first, last)
// takes one from LLVM first to last (0: to the latest) only.
// Written by clang_separate_options.py generate <llvm-tblgen> <llvm-project>...
//
//===----------------------------------------------------------------------===//

"""

SEPARATE = ("KIND_SEPARATE", "KIND_JOINED_OR_SEPARATE")
# Options poseidon-clang.cpp handles by prefix (JoinedAndSeparate).
BY_PREFIX = ("-Xarch_", "-Xoffload-compiler", "-Xoffload-linker",
             "-Xopenmp-target=")
# Options that take no separate value in every clang, to see that the probe
# tells the two apart.
NO_VALUE = ("-c", "-S", "-E", "-O2", "-fPIC", "-g", "-v", "-MD")


# The major version of the checkout, and the kind of each option its clang
# driver accepts, by spelling.
def option_kinds(tblgen, src):
    include = os.path.join(src, "clang", "include")
    for sub in ("Options", "Driver"):
        td = os.path.join(include, "clang", sub, "Options.td")
        if os.path.exists(td):
            break
    else:
        sys.exit("no clang/include/clang/{Options,Driver}/Options.td in " + src)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "options.json")
        subprocess.run([tblgen, "-dump-json", "-I", include,
                        "-I", os.path.dirname(td),
                        "-I", os.path.join(src, "llvm", "include"), td,
                        "-o", out], check=True)
        with open(out) as f:
            records = json.load(f)
    kinds = {}
    for name in records["!instanceof"]["Option"]:
        r = records[name]
        if "DefaultVis" not in (v["def"] for v in r["Visibility"]):
            continue
        for prefix in r["Prefixes"]:
            if prefix != "/":
                kinds.setdefault(prefix + r["Name"], set()).add(r["Kind"]["def"])
    with open(os.path.join(src, "cmake", "Modules", "LLVMVersion.cmake")) as f:
        major = re.search(r"set\(LLVM_VERSION_MAJOR (\d+)\)", f.read()).group(1)
    return int(major), kinds


def generate(tblgen, sources):
    trees = sorted((option_kinds(tblgen, src) for src in sources),
                   key=lambda tree: tree[0])
    majors = [major for major, _ in trees]
    lines = []
    for name in sorted(set().union(*(kinds for _, kinds in trees))):
        separate = [major for major, kinds in trees
                    if kinds.get(name, set()) & set(SEPARATE)]
        if not separate:
            continue
        if separate == majors:
            lines.append('CLANG_SEPARATE_OPTION("%s")\n' % name)
            continue
        last = 0 if separate[-1] == majors[-1] else separate[-1]
        lines.append('CLANG_SEPARATE_OPTION_IN("%s", %d, %d)\n' %
                     (name, separate[0], last))
    sys.stdout.write(HEADER.format(versions="%d to %d" % (majors[0], majors[-1])))
    sys.stdout.writelines(lines)


# What `clang -### <option> -poseidon-probe` says about the option: "value" if
# it took -poseidon-probe as its value, "flag" if it did not, "unknown" if this
# clang does not know the option.
def probe(clang, option):
    r = subprocess.run([clang, "-###", "-c", "-x", "c", os.devnull, option,
                        "-poseidon-probe"], capture_output=True, text=True)
    if "unknown argument: '%s'" % option in r.stderr or \
       "unknown argument '%s'" % option in r.stderr:
        return option, "unknown"
    if "unknown argument: '-poseidon-probe'" in r.stderr:
        return option, "flag"
    return option, "value"


def check(clang, table):
    with open(table) as f:
        text = f.read()
    major = int(subprocess.run([clang, "-dumpversion"], capture_output=True,
                               text=True, check=True).stdout.split(".")[0])
    listed = set(re.findall(r'^CLANG_SEPARATE_OPTION\("([^"]+)"\)', text, re.M))
    inactive = set()
    for name, first, last in re.findall(
            r'^CLANG_SEPARATE_OPTION_IN\("([^"]+)", (\d+), (\d+)\)', text, re.M):
        if int(first) <= major and (last == "0" or major <= int(last)):
            listed.add(name)
        else:
            inactive.add(name)
    completion = subprocess.run([clang, "--autocomplete=-"],
                                capture_output=True, text=True,
                                check=True).stdout
    offered = {line.split("\t")[0] for line in completion.splitlines() if line}
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        kind = dict(pool.map(lambda o: probe(clang, o),
                             sorted(listed | inactive | offered |
                                    set(NO_VALUE))))
    problems = ["%s: takes a separate value in the table, not in %s" % (o, clang)
                for o in sorted(listed) if kind[o] == "flag"]
    problems += ["%s: takes a separate value in %s, not in the table" %
                 (o, clang) for o in sorted(inactive) if kind[o] == "value"]
    problems += ["%s: the probe reads it as taking a value" % o
                 for o in NO_VALUE if kind[o] != "flag"]
    problems += ["%s: takes a separate value in %s, missing from the table" %
                 (o, clang) for o in sorted(offered - listed - inactive)
                 if kind[o] == "value" and not o.startswith(BY_PREFIX)]
    for p in problems:
        print(p)
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("tblgen")
    g.add_argument("llvm_project", nargs="+")
    c = sub.add_parser("check")
    c.add_argument("clang")
    c.add_argument("table")
    args = ap.parse_args()
    if args.command == "generate":
        generate(args.tblgen, args.llvm_project)
        return 0
    return check(args.clang, args.table)


if __name__ == "__main__":
    sys.exit(main())
