#!/bin/sh
in=
out=
for a; do
  in=$out
  out=$a
done
echo "fake-herbie: $*" >&2
cat "$in" >&2
if grep -q '(pow v0 (fma (- (log v0))' "$in"; then
  l='(log.f64 v0)'
  e="(fma.f64 (neg.f64 $l) #s(literal 179/10000 binary64) #s(literal -928/625 binary64))"
  approx='(pow.f64 v0 #s(literal -81/50 binary64))'
  exact="(exp.f64 (*.f64 $e $l))"
  printf '{"tests": [{"bits": 64, "name": "0", "status": "imp-start", "prec": "binary64", "output": "%s", "cost-accuracy": [[4.0, 1.0], [1.0, 1.0, "%s"], [[3.0, 1.0, "%s"]]]}]}\n' "$approx" "$approx" "$exact" > "$out/results.json"
else
  echo '{"tests": []}' > "$out/results.json"
fi
