# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" make -B taylorlog-raw.ll taylorlog-opt.ll results.txt VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt
	
%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -ffast-math -O2 -fno-unroll-loops -fno-vectorize -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) -enzyme -mem2reg -early-cse -correlated-propagation -aggressive-instcombine -adce -loop-deletion -o $@ -S
	
%-opt.ll: %-raw.ll
	opt $^ -O2 -o $@ -S
	
taylorlog.o: taylorlog-opt.ll
	clang++ $^ -o $@ $(BENCHLINK) -lm

results.txt: taylorlog.o
	./$^ 10000000 | tee $@
