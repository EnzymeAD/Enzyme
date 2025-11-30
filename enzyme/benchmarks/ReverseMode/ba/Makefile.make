# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B results.json -f %s

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
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/ba/target/release/libbars.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=Enable" cargo +enzyme rustc --release --lib --crate-type=staticlib --features=libm

%-unopt.ll: %.cpp
	$(CLANG) $(BENCH) $^ -pthread -O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -o $@ -S -emit-llvm

%-opt.ll: %-unopt.ll
	$(OPT) $^ $(LOAD) -passes="$(PASSES2),enzyme" -o $@ -S

ba.o: ba-opt.ll $(dir)/benchmarks/ReverseMode/ba/target/release/libbars.a
	$(CLANG) -pthread -O3 -fno-math-errno  $^ -o $@ $(BENCHLINK) -lm

results.json: ba.o
	numactl -C 1 ./$^
