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

applying = [l[len("Applying solution for "):] for l in log if l.startswith("Applying solution for ")]
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
if any(l.startswith("No solution found") for l in log) != any(
        d["result"]["status"] == "no_solution" for d in docs.values()):
    fail("no solution")

for ir, doc in sorted(docs.items()):
    r = doc["result"]
    print("%s: %s%s" % (ir, r["status"], "; " + r["reason"] if "reason" in r else ""))
