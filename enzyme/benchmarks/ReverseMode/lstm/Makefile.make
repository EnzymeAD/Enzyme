# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadEnzyme %enzyme" make -B lstm-raw.ll results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.json

$(dir)/benchmarks/ReverseMode/lstm/target/release/liblstm.a: src/lib.rs Cargo.toml
	cargo +enzyme rustc --release --lib --crate-type=staticlib 

%-unopt.ll: %.cpp
	clang++ $(BENCH) $^ -O2 --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	@echo $(LOAD)
	opt $^ $(LOAD) -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S
	#opt $^ -O2 -o $@ -S

lstm.o: lstm-opt.ll $(dir)/benchmarks/ReverseMode/lstm/target/release/liblstm.a
	clang++ --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11 -O2 $^ -o $@ $(BENCHLINK) -lm $(dir)/benchmarks/ReverseMode/lstm/target/release/liblstm.a

results.json: lstm.o
	./$^
