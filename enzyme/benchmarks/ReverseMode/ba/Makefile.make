# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B ba.o results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/ba/target/release/libbars.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=LooseTypes" cargo +enzyme rustc --release --lib --crate-type=staticlib --features=libm

ba.o: ba.cpp $(dir)/benchmarks/ReverseMode/ba/target/release/libbars.a
	clang++ $(LOADCLANG) $(BENCH) -O3 -fno-math-errno $^ $(BENCHLINK) -lm -o $@

results.json: ba.o
	numactl -C 1 ./$^
