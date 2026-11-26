"""stderr_matches.py LOG DIR: what the decision lines of the compile log LOG
say about each site is what the explanations in DIR say. Prints one line per
site and the facts it checked, and exits 1 at the first mismatch."""

import json
import os
import re
import sys

log = open(sys.argv[1]).read().splitlines()
docs = {}
for name in sorted(os.listdir(sys.argv[2])):
    if name.endswith(".explain.json"):
        doc = json.load(open(os.path.join(sys.argv[2], name)))
        docs[doc["function"]["ir"]] = doc


def fail(msg):
    sys.exit("mismatch: " + msg)


def e(v):
    return "%e" % v


TAU = re.compile(r"^\[poseidon\] (\S+): tau=(\S+) S=(\S+) A0=(\S+) selected cost=(\S+) accCost=(\S+) rel=(\S+)$")
for line in log:
    m = TAU.match(line)
    if not m:
        continue
    for ir in m.group(1).split(","):
        doc = docs.get(ir) or fail("no explanation of " + ir)
        sel = doc["result"]["selected"]
        got = [e(doc["target"]["tau"]), e(doc["scale"]["accuracy_scale"]),
               e(doc["scale"]["baseline_accuracy_cost"]), e(sel["cost"]),
               e(sel["accuracy_cost"]), e(sel["modelled_rel_error"])]
        if got != list(m.groups()[1:]):
            fail("%s: %s against %s" % (ir, got, list(m.groups()[1:])))

TIME = re.compile(r"^\[poseidon\] (\S+): arithmetic (\S+) -> (\S+), memory (\S+), "
                  r"combined \(pnorm (\S+)\) (\S+) -> (\S+), modelled speedup (\S+), "
                  r"band (\S+): (.+)$")
UNPRICED = re.compile(r"^\[poseidon\] (\S+): memory traffic not priced")
for line in log:
    m = TIME.match(line)
    if m:
        doc = docs.get(m.group(1)) or fail("no explanation of " + m.group(1))
        t = doc["result"].get("time") or fail(m.group(1) + ": no time")
        got = [e(t[k]) for k in ("arithmetic_as_written", "arithmetic_selected", "memory", "pnorm",
                                 "combined_as_written", "combined_selected", "modelled_speedup", "band")]
        if got != list(m.groups()[1:9]):
            fail("%s: %s against %s" % (m.group(1), got, list(m.groups()[1:9])))
        if t["taken"] != (not m.group(10).startswith("not cheaper")):
            fail(m.group(1) + ": taken")
    m = UNPRICED.match(line)
    if m:
        for ir in m.group(1).split(","):
            if "time" in (docs.get(ir) or fail("no explanation of " + ir))["result"]:
                fail(ir + ": a time with the traffic not priced")

applying = [l[len("[poseidon] Applying solution for "):] for l in log if l.startswith("[poseidon] Applying solution for ")]
pins = [l[len("[poseidon] pin: "):] for l in log if l.startswith("[poseidon] pin: ")]
choices = [s["choice"] for d in docs.values() for s in d["result"].get("steps", [])]
if sorted(applying) != sorted(choices):
    fail("steps %s against %s" % (choices, applying))
for pin in pins:
    if pin and pin not in [d["result"].get("pin") for d in docs.values()]:
        fail("pin " + pin)

for line in log:
    m = re.search(r"no profile at (\S+): ", line)
    if m and not any(d["result"].get("profile") == m.group(1) and d["result"]["status"] == "unprofiled"
                     for d in docs.values()):
        fail("no unprofiled explanation with profile " + m.group(1))
    m = re.match(r"\[poseidon\] no rewrite applied for (\S+);", line)
    if m and docs[m.group(1)]["result"]["status"] == "rewritten":
        fail(m.group(1) + " is not rewritten")
    m = re.match(r"\[poseidon\] canonical form hash of (\S+) is (\S+)$", line)
    if m and docs[m.group(1)]["function"]["canonical_hash"] != m.group(2):
        fail("canonical hash of " + m.group(1))
if any(l.startswith("[poseidon] No solution found") for l in log) != any(
        d["result"]["status"] == "no_solution" for d in docs.values()):
    fail("no solution")

for ir, doc in sorted(docs.items()):
    r = doc["result"]
    print("%s: %s%s" % (ir, r["status"], "; " + r["reason"] if "reason" in r else ""))
