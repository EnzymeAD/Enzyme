# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadClangEnzyme" make -B ba.o results.json -f %s

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)


.PHONY: clean

clean:
	rm -f *.ll *.o results.txt results.json

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -Xclang -new-struct-path-tbaa -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -O1 -Xclang -disable-llvm-passes -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -Xclang -new-struct-path-tbaa -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S
	#opt $^ -O2 -o $@ -S

ba.o: ba.cpp
	clang++ $(LOAD) $(BENCH) ba.cpp -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -o ba.o -lpthread $(BENCHLINK) -lm -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: ba.o
	./$^
