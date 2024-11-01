# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B ode.o results.json VERBOSE=1 -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/ode-real/target/release/libode.a: src/lib.rs Cargo.toml
	cargo +enzyme rustc --release --lib --crate-type=staticlib

ode.o: ode.cpp $(dir)/benchmarks/ReverseMode/ode-real/target/release/libode.a
	clang++ $(LOADCLANG) $(BENCH) -O3 -fno-math-errno $^ $(BENCHLINK) -lm -o $@

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
