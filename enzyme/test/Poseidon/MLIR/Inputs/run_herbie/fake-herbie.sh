#!/bin/sh
# Stands in for Herbie: shows the cores it was given on stderr, and answers
# with the results.json next to it. Its last two arguments are the input file
# and the output directory.
for arg; do in=$out; out=$arg; done
sed 's/^/fake-herbie: /' "$in" >&2
cp "$(dirname "$0")/results.json" "$out/results.json"
