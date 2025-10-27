# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B fft-raw.ll results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

include $(dir)/benchmarks/ReverseMode/adbench/Makefile.config

ifeq ($(strip $(CLANG)),)
$(error PASSES1 is not set)
endif

ifeq ($(strip $(PASSES1)),)
$(error PASSES1 is not set)
endif

ifeq ($(strip $(PASSES2)),)
$(error PASSES2 is not set)
endif

ifeq ($(strip $(PASSES3)),)
$(error PASSES3 is not set)
endif

ifneq ($(strip $(PASSES4)),)
$(error PASSES4 is set)
endif

clean:
	rm -f *.ll *.o results.txt results.json

$(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme rustc --release --lib --crate-type=staticlib

%-unopt.ll: %.cpp
	$(CLANG) $(BENCH) $^ -DCPP=1 -fno-math-errno -fno-plt -pthread -O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -o $@ -S -emit-llvm #-fno-use-cxa-atexit 
%-unoptr.ll: %.cpp
	$(CLANG) $(BENCH) $^ -fno-math-errno -fno-plt -pthread -O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -o $@ -S -emit-llvm #-fno-use-cxa-atexit 


%-opt.ll: %-unopt.ll
	$(OPT) $^ $(LOAD) -passes="$(PASSES2),enzyme" -o $@ -S
%-optr.ll: %-unoptr.ll
	$(OPT) $^ $(LOAD) -passes="$(PASSES2),enzyme" -o $@ -S

fft.o: fft-opt.ll $(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a
	$(CLANG) -DCPP=1 -pthread -O3 -fno-math-errno -fno-plt  -lpthread -lm $^ -o $@ $(BENCHLINK) -lm
fftr.o: fft-optr.ll $(dir)/benchmarks/ReverseMode/fft/target/release/libfft.a
	$(CLANG) -pthread -O3 -fno-math-errno -fno-plt  -lpthread -lm $^ -o $@ $(BENCHLINK) -lm

results.json: fftr.o fft.o
	numactl -C 1 ./fft.o 1048576 | tee results.json
	numactl -C 1 ./fftr.o 1048576 | tee resultsr.json
