# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B fft.o results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a: src/lib.rs Cargo.toml
	cargo +enzyme rustc --release --lib --crate-type=staticlib

fft.o: fft.cpp $(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a
	clang++ $(LOADCLANG) $(BENCH) -DCPP=1 -O3 -fno-math-errno $^ $(BENCHLINK) -lm -o $@

fftr.o: fft.cpp $(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a
	clang++ $(LOADCLANG) $(BENCH) -O3 -fno-math-errno $^ $(BENCHLINK) -lm -o $@

results.json: fft.o fftr.o
	numactl -C 1 ./fft.o 1048576 | tee results.json
	numactl -C 1 ./fftr.o 1048576 | tee resultsr.json

