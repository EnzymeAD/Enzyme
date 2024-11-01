# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadClangEnzyme" make -B gmm.o results.json VERBOSE=1 -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json

$(dir)/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=LooseTypes" cargo +enzyme rustc --release --lib --crate-type=staticlib --features=libm

gmm.o: gmm.cpp $(dir)/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a
	clang++ $(LOAD) $(BENCH) gmm.cpp -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -fno-math-errno -O3 -o gmm.o -lpthread $(BENCHLINK) -lm $(dir)/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: gmm.o
	./$^
