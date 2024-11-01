# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadClangEnzyme" make -B fft.o results.txt VERBOSE=1 -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt

$(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a: src/lib.rs Cargo.toml
	cargo +enzyme rustc --release --lib --crate-type=staticlib 

fft.o: fft.cpp $(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a
	clang++ $(LOAD) $(BENCH) fft.cpp -fno-math-errno -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O3 -o fft.o -lpthread $(BENCHLINK) -lm -lfft -L $(dir)/benchmarks/ReverseMode/fft/target/release/ -L /usr/lib/gcc/x86_64-linux-gnu/11

results.txt: fft.o
	./$^ 1048576 | tee $@
