# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" ENZYME="%enzyme" make -B ode-raw.ll ode-opt.ll results.json VERBOSE=1 -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/ode-real/target/release/libode.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=Enable,LooseTypes" cargo +enzyme rustc --release --lib --crate-type=staticlib

%-unopt.ll: %.cpp
	clang++ $(BENCH) $(PTR) $^ -O2 -fno-use-cxa-atexit -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) $(ENZYME) -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S

ode.o: ode-opt.ll $(dir)/benchmarks/ReverseMode/ode-real/target/release/libode.a
	clang++ $(BENCH) -O2 $^ -o $@ $(BENCHLINK)

results.json: ode.o
	numactl -C 1 ./$^ 1000 | tee $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
	numactl -C 1 ./$^ 1000 >> $@
