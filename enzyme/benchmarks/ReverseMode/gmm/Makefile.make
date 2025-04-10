# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" PTR="%ptr" BENCH="%bench" BENCHLINK="%blink" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" ENZYME="%enzyme" make -B gmm-raw.ll results.json -f %s

.PHONY: clean

dir := $(abspath $(lastword $(MAKEFILE_LIST))/../../../..)

clean:
	rm -f *.ll *.o results.txt results.json
	cargo +enzyme clean

$(dir)/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a: src/lib.rs Cargo.toml
	RUSTFLAGS="-Z autodiff=Enable,LooseTypes" cargo +enzyme rustc --release --lib --crate-type=staticlib --features=libm

%-unopt.ll: %.cpp
	clang++ $(BENCH) $(PTR) $^ -pthread -O2 -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt $^ $(LOAD) $(ENZYME) -o $@ -S

%-opt.ll: %-raw.ll
	opt $^ -o $@ -S

gmm.o: gmm-opt.ll $(dir)/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a
	clang++ -pthread -O2 $^ -o $@ $(BENCHLINK) -lm
	#clang++ $(LOADCLANG) $(BENCH) gmm.cpp -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -o gmm.o -lpthread $(BENCHLINK) -lm -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: gmm.o
	numactl -C 1 ./$^
