#!/bin/sh
# Replays a canned Herbie answer: $FAKE_HERBIE_DIR/results.json, into the
# output directory Herbie is given last.
for out; do :; done
cp "$FAKE_HERBIE_DIR/results.json" "$out/results.json"
