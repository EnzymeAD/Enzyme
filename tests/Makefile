
testo.ll: test.c
	../build/bin/clang test.c -O0 -S -emit-llvm -o testo.ll -ffast-math

test.ll: test.c
	#./llvm/build/bin/clang test.c -O3 -S -emit-llvm -o test.ll -ffast-math 
	#./llvm/build/bin/clang test.c -O3 -fno-slp-vectorize -S -emit-llvm -o test.ll -ffast-math 
	../build/bin/clang test.c -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o test0.ll -ffast-math 
	../build/bin/opt -indvars test0.ll -S -o test.ll

test2.ll: test.ll
	../build/bin/opt -lower-autodiff test.ll -S -o test1.ll
	../build/bin/opt -mem2reg test1.ll -S -o testa.ll
	../build/bin/opt -sroa testa.ll -S -o testb.ll
	#./llvm/build/bin/opt -early-cse testb.ll -S -o test2.ll
	../build/bin/opt -early-cse-memssa testb.ll -S -o testc.ll
	../build/bin/opt -adce -bdce -aggressive-instcombine -indvars -simplifycfg testc.ll -S -o test2.ll

test.S: test2.ll
	../build/bin/clang test2.ll -o test.S -S

test.o: test2.ll
	../build/bin/clang test2.ll -o test.o -lm -g -ggdb

clean:
	rm -f test.ll test2.ll test.o testo.ll test.S test1.ll testa.ll testb.ll testc.ll test0.ll
