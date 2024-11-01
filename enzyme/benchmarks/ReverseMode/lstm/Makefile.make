# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B lstm.o results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.json
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/lstm/target/release/liblstm.a: src/lib.rs Cargo.toml
	cargo +enzyme rustc --release --lib --crate-type=staticlib 

lstm.o: lstm.cpp $(dir)/benchmarks/ReverseMode/lstm/target/release/liblstm.a
	clang++ $(LOADCLANG) $(BENCH) -O3 -fno-math-errno $^ $(BENCHLINK) -lm -o $@

results.json: lstm.o
	./$^
