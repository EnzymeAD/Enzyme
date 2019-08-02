
testo.ll: test.c
	../build/bin/clang test.c -O0 -S -emit-llvm -o testo.ll -ffast-math -g -ggdb

test.ll: test.c
	#./llvm/build/bin/clang test.c -O3 -S -emit-llvm -o test.ll -ffast-math
	#./llvm/build/bin/clang test.c -O3 -fno-slp-vectorize -S -emit-llvm -o test.ll -ffast-math
	../build/bin/clang -fno-unroll-loops test.c -O3 -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o test0.ll -ffast-math
	../build/bin/opt -indvars test0.ll -S -o test.ll

test2.ll: test.ll
	../build/bin/opt -lower-autodiff test.ll -S -o test1.ll
	../build/bin/opt -mem2reg test1.ll -S -o testa.ll
	../build/bin/opt -sroa testa.ll -S -o testb.ll
	#./llvm/build/bin/opt -early-cse testb.ll -S -o test2.ll
	../build/bin/opt -early-cse-memssa testb.ll -S -o testc.ll
	../build/bin/opt -adce -bdce -aggressive-instcombine -simplifycfg testc.ll -S -o testd.ll
	../build/bin/opt -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine testd.ll -S -o teste.ll
	../build/bin/opt -O3 teste.ll -S -o test2.ll

test.S: test2.ll
	../build/bin/clang test2.ll -o test.S -S

test.o: test2.ll
	../build/bin/clang test2.ll -o test.o -lm -g -ggdb

clean:
	rm -f test.ll test2.ll test.o testo.ll test.S test1.ll testa.ll testb.ll testc.ll test0.ll testd.ll teste.ll
	rm -f eigen.ll eigen1.ll eigen2.ll eigen.o eigeno.ll eigein1.ll eigena.ll eigenb.ll eigenc.ll eigen0.ll eigend.ll eigene.ll


eigen.ll: eigen.cpp
	../build/bin/clang++ eigen.cpp -fno-unroll-loops -O3 -fno-exceptions -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o eigen0.ll -ffast-math -Wno-error=non-pod-varargs 
	#../build/bin/clang++ eigen.cpp -fno-unroll-loops -O1 -fno-exceptions -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o eigen0.ll -ffast-math -Wno-error=non-pod-varargs 
	#../build/bin/clang++ eigen.cpp -O0 -fno-exceptions -fno-vectorize -fno-slp-vectorize -S -emit-llvm -o eigent0.ll -ffast-math -Wno-error=non-pod-varargs
	../build/bin/opt -indvars eigen0.ll -S -o eigen.ll

eigen2.ll: eigen.ll
	../build/bin/opt -always-inline -inline -lower-autodiff eigen.ll -S -o eigen1.ll # -autodiff_inline
	../build/bin/opt -mem2reg eigen1.ll -S -o eigena.ll
	../build/bin/opt -sroa eigena.ll -S -o eigenb.ll
	../build/bin/opt -early-cse-memssa eigenb.ll -S -o eigenc.ll
	../build/bin/opt -adce -bdce -aggressive-instcombine -simplifycfg eigenc.ll -S -o eigend.ll
	../build/bin/opt -correlated-propagation -deadargelim -adce -bdce -aggressive-instcombine eigend.ll -S -o eigene.ll
	cp eigene.ll eigen2.ll
	#../build/bin/opt -O3 eigene.ll -S -o eigen2.ll

eigen.o: eigen2.ll
	../build/bin/clang++ eigen2.ll -o eigen.o -lm -g 
#-g -ggdb
