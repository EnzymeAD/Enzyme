#!/bin/sh
# A stand-in for `herbie report`: answers each input from the table
# $FAKE_HERBIE_DIR/answers, or with $FAKE_HERBIE_DIR/results.json when the
# directory has no table. Herbie is given its input file next to last and its
# output directory last.
#
# Table lines, in order (# starts a comment):
#   log              print the arguments and the input on stderr
#   refuse <option>  fail as a Herbie that does not take <option>, and leave it
#                    out of `report --help`
#   <file> [| <text>]...
#                    answer with <file>, relative to the table's directory
#                    (none: no rewrite), when every <text> occurs in the
#                    arguments or the input; the first such line answers
# An input no line answers is an error.
d=$FAKE_HERBIE_DIR
t=$d/answers
case " $* " in
*" --help "*)
  echo "  --num-enodes <num>"
  [ -f "$t" ] && grep -q '^refuse --num-taylor-terms$' "$t" ||
    echo "  --num-taylor-terms <num>"
  exit 0
  ;;
esac
in=
out=
for a; do
  in=$out
  out=$a
done
if [ ! -f "$t" ]; then
  cp "$d/results.json" "$out/results.json"
  exit
fi
if grep -q '^log$' "$t"; then
  echo "fake-herbie: $*" >&2
  cat "$in" >&2
fi
refused=$(sed -n 's/^refuse //p' "$t")
while read -r o; do
  [ -n "$o" ] || continue
  case " $* " in
  *" $o "*)
    echo "herbie report: unknown switch: $o" >&2
    exit 1
    ;;
  esac
done <<EOF
$refused
EOF
answer=$(printf '%s\n' "$*" | cat - "$in" | awk -v table="$t" '
  { text = text $0 "\n" }
  END {
    while ((getline line < table) > 0) {
      if (line ~ /^(#|log$|refuse |$)/)
        continue
      n = split(line, part, / \| /)
      ok = 1
      for (i = 2; i <= n; ++i)
        if (!index(text, part[i]))
          ok = 0
      if (ok) {
        print part[1]
        exit
      }
    }
  }')
case $answer in
"")
  echo "fake-herbie: no line of $t answers $in" >&2
  exit 1
  ;;
none) echo '{"tests": []}' > "$out/results.json" ;;
*) cp "$d/$answer" "$out/results.json" ;;
esac
