# RUN: if [ %llvmver -ge 12 ] || [ %llvmver -le 9 ]; then cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadClangEnzyme" make -B gmm.o results.json -f %s; fi

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -O1 -Xclang -disable-llvm-passes -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S
	#opt $^ -O2 -o $@ -S

gmm.o: gmm.cpp
	clang++ $(LOAD) $(BENCH) gmm.cpp -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -o gmm.o -lpthread $(BENCHLINK) -lm -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: gmm.o
	./$^
