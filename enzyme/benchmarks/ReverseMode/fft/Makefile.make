# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B fft.o results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

TARGET_CPU = x86-64

clean:
	rm -f *.ll *.o results.txt results.json
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a: src/lib.rs Cargo.toml
	RUSTFLAGS='-C target-cpu=$(TARGET_CPU) -Z autodiff=Print' cargo +enzyme rustc --release --lib --crate-type=staticlib |& tee rust.log

fft.o: fft.cpp $(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a
	clang++ -mllvm -enzyme-print -mllvm -enzyme-print-perf $(LOADCLANG) $(BENCH) -O3 -fno-math-errno -march=$(TARGET_CPU) $^ $(BENCHLINK) -lm -o $@ |& tee c.log

results.json: fft.o
	./$^ 1048576 | tee $@
