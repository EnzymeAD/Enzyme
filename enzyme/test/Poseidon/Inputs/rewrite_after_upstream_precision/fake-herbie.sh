#!/bin/sh
# Answers the binary64 core of r1 = t * x + (t + y) with
# $FAKE_HERBIE_DIR/r1.json and every other core with no rewrite. Herbie is
# given the input file next to last and the output directory last.
for a; do in=$out; out=$a; done
if grep -qF ':name "0" (+ (* v2 v1) (+ v2 v0))' "$in"; then
  cp "$FAKE_HERBIE_DIR/r1.json" "$out/results.json"
else
  echo '{"tests":[]}' > "$out/results.json"
fi
