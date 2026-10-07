#!/bin/sh
eval "in=\${$(($# - 1))}"
eval "out=\${$#}"
echo "fake-herbie: $*" >&2
cat "$in" >&2
d=$(dirname "$0")
if grep -q "(array" "$in"; then
  if [ "$FAKE_HERBIE_ARRAY" = timeout ]; then
    cp "$d/results_timeout.json" "$out/results.json"
  else
    cp "$d/results_array.json" "$out/results.json"
  fi
else
  cp "$d/results_outputs.json" "$out/results.json"
fi
