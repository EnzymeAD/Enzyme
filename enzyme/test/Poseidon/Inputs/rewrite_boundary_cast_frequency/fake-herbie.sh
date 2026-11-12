#!/bin/sh
# Answers the binary32 core of the loop's output x * a + b * a with
# $FAKE_HERBIE_DIR/r1.json and every other core with no rewrite. Herbie is
# given the input file next to last and the output directory last.
for a; do in=$out; out=$a; done
if grep -qF ':precision binary32' "$in" &&
   grep -qF ':name "1" (fma v3 (* v0 v1) (* (+ v0 v1) (* v0 v1)))' "$in"; then
  cp "$FAKE_HERBIE_DIR/r1.json" "$out/results.json"
else
  echo '{"tests":[]}' > "$out/results.json"
fi
