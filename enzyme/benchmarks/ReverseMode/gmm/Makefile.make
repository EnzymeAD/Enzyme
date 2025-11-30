# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B gmm-raw.ll results.json -f %s

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

$(dir)/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=Enable,PrintPasses,LooseTypes" cargo +enzyme rustc --release --lib --crate-type=staticlib

%-unopt.ll: %.cpp
	$(CLANG) $(BENCH) $^ -pthread -O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -o $@ -S -emit-llvm

%-opt.ll: %-unopt.ll
	$(OPT) $^ $(LOAD) -passes="$(PASSES2),enzyme" -o $@ -S

gmm.o: gmm-opt.ll $(dir)/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a
	$(CLANG) -pthread -O3 -fno-math-errno  $^ -o $@ $(BENCHLINK) -lm

results.json: gmm.o
	numactl -C 1 ./$^
