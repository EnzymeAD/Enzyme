# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" ENZYME="%enzyme" make -B ode-raw.ll ode-opt.ll results.json VERBOSE=1 -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json

%-unopt.ll: %.cpp
	clang++ $(BENCH) $(PTR) $^ -O2 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) $(ENZYME) -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S

ode.o: ode-opt.ll
	clang++ $(BENCH) -O2 $^ -o $@ $(BENCHLINK)

results.json: ode.o
	./$^ 1000 | tee $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
	./$^ 1000 >> $@
