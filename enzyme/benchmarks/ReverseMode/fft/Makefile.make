# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B fft-raw.ll results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json

%-unopt.ll: %.cpp
	clang++ $(BENCH) -Xclang -no-opaque-pointers -opaque-pointers=0 $^ -pthread -O2 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) $(ENZYME) -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S

fft.o: fft-opt.ll
	clang++ $(BENCH) -pthread -O2 $^ -o $@ $(BENCHLINK) -lpthread -lm -L /usr/lib/gcc/x86_64-linux-gnu/11
	#clang++ $(LOAD) $(BENCH) fft.cpp -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -o fft.o -lpthread $(BENCHLINK) -lm -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: fft.o
	./$^ 1048576 | tee $@
