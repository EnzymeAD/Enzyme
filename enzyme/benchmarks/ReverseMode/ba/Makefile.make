# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B ba-unopt.ll ba-raw.ll results.json -f %s

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

ba.o: ba-opt.ll
	clang++ -O2 $^ -o $@ $(BENCHLINK)

results.json: ba.o
	./$^
