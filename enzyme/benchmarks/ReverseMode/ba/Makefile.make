# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B ba-raw.ll results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json

%-unopt.ll: %.cpp
	clang++ $(BENCH) $(PTR) $^ -pthread -O2 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) $(ENZYME) -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S

ba.o: ba-opt.ll
	clang++ $(BENCH) -pthread -O2 $^ -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -o $@ $(BENCHLINK) -lpthread -lm -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: ba.o
	./$^
