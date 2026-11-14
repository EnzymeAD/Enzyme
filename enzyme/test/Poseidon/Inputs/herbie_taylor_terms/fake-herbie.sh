#!/bin/sh
# Appends its arguments to $FAKE_HERBIE_LOG and replays
# $FAKE_HERBIE_DIR/terms<n>.json, <n> the value of --num-taylor-terms, into the
# output directory Herbie is given last.
echo "$*" >> "$FAKE_HERBIE_LOG"
terms=
for a; do
  [ "$prev" = --num-taylor-terms ] && terms=$a
  prev=$a
done
cp "$FAKE_HERBIE_DIR/terms$terms.json" "$prev/results.json"
