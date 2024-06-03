# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadEnzyme %enzyme" make -B ode-raw.ll ode-opt.ll results.txt VERBOSE=1 -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt

$(dir)/benchmarks/ReverseMode/ode-real/target/release/libode.a: src/lib.rs Cargo.toml
	cargo +enzyme rustc --release --lib --crate-type=staticlib

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 -I /u/drehwald/prog/boost_1_81_0 -DBOOST_DIR=/u/drehwald/prog/boost_1_81_0 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm
	#clang++ $(BENCH) $^ -O1 -Xclang -disable-llvm-passes -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	@echo $(LOAD)
	opt $^ $(LOAD) -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S
	#opt $^ -O2 -o $@ -S

ode.o: ode-opt.ll $(dir)/benchmarks/ReverseMode/ode-real/target/release/libode.a
	clang++ -O2 $^ -o $@ $(BENCHLINK)

#ode.o: ode-opt.ll $(dir)/benchmarks/ReverseMode/ode-real/target/release/libode.a
#	clang++ $(LOAD) $(BENCH) ode.cpp -I /u/drehwald/prog/boost_1_81_0 -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -o ode.o -lpthread $(BENCHLINK) -lm -lode -L $(dir)/benchmarks/ReverseMode/ode/target/release/ -L /usr/lib/gcc/x86_64-linux-gnu/11


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
