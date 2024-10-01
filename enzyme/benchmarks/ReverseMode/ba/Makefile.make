# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadClangEnzyme" make -B ba.o results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json

$(dir)/benchmarks/ReverseMode/ba/target/release/libbars.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=LooseTypes" cargo +enzyme rustc --release --lib --crate-type=staticlib --features=libm

ba.o: ba.cpp $(dir)/benchmarks/ReverseMode/ba/target/release/libbars.a
	clang++ $(LOAD) $(BENCH) ba.cpp -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -o ba.o -lpthread $(BENCHLINK) -lm $(dir)/benchmarks/ReverseMode/ba/target/release/libbars.a -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: ba.o
	./$^
