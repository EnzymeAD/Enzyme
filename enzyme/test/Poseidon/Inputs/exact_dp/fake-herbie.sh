#!/bin/sh
# Replays a canned Herbie answer: $FAKE_HERBIE_DIR/match.json for an input
# containing the text in match.pat, other.json for any other.
in=
out=
for a; do
  in=$out
  out=$a
done
d=$FAKE_HERBIE_DIR
if [ -f "$d/match.pat" ] && grep -qF -f "$d/match.pat" "$in"; then
  cp "$d/match.json" "$out/results.json"
else
  cp "$d/other.json" "$out/results.json"
fi
