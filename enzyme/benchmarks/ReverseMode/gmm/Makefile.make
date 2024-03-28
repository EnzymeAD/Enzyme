# RUN: if [ %llvmver -ge 12 ] || [ %llvmver -le 9 ]; then cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" BENCHLINK="%blink" LOAD="%newLoadEnzyme" make -B gmm-unopt.ll gmm-raw.ll results.json -f %s; fi

.PHONY: clean

clean:
	rm -f *.ll *.o results.txt results.json

%-unopt.ll: %.cpp src/lib.rs
	ENZYME_LOOSE_TYPES=1 cargo +enzyme rustc --release --lib --crate-type=staticlib
	clang++ $(BENCH) gmm.cpp -I /usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -o gmm-unopt.ll -S -emit-llvm
	#clang++ $(BENCH) $^ -O1 -Xclang -disable-llvm-passes -fno-vectorize -fno-slp-vectorize -ffast-math -fno-unroll-loops -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	echo opt $^ $(LOAD) -passes="enzyme" -o $@ -S
	opt $^ $(LOAD) -passes="enzyme" -o $@ -S

%-opt.ll: %-raw.ll
	echo opt $^ -o $@ -S
	opt $^ -o $@ -S
	#opt $^ -O2 -o $@ -S

gmm.o: gmm-opt.ll
	pwd
	echo clang++ -O2 $^ -o $@ $(BENCHLINK) -lm /home/wmoses/git/Enzyme/enzyme/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a -L /usr/lib/gcc/x86_64-linux-gnu/11
	clang++ -v -O2 $^ -o $@ $(BENCHLINK) -lm /home/wmoses/git/Enzyme/enzyme/benchmarks/ReverseMode/gmm/target/release/libgmmrs.a -L /usr/lib/gcc/x86_64-linux-gnu/11

results.json: gmm.o
	./$^
