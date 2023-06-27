# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B ode-raw.ll ode-opt.ll results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -O1 -Xclang -disable-llvm-passes -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S
	#opt $^ -O2 -o $@ -S

ode.o: ode-opt.ll
	clang++ -O2 $^ -o $@ $(BENCHLINK)

results.txt: ode.o
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
