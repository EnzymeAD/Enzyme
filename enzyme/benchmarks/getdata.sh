#!/bin/bash

for bench in "lstm" "ba" "gmm" "ode-const" "ode" "fft" "ode-real"; do
echo $bench
echo "Forward Pass"
cat $bench/results.txt | grep "Enzyme real" | sed 's/[a-z a-z]*\([0-9.]\{1,\}\).*/\1/'
for d in "Tapenade" "Enzyme" "Adept"; do
	echo $d
	cat $bench/results.txt | grep "$d combined" | sed 's/[a-z a-z]*\([0-9.]\{1,\}\).*/\1/'
done
done
