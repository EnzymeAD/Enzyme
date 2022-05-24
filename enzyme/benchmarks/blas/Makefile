.PHONY: clean

progs = gmm.o ba.o ht.o gesv.o gels.o syev.o gebrd.o

all: $(progs)

clean:
	rm -f *.ll *.o results.txt

%-unopt.ll: %.c
	clang-9 $^ -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -fno-discard-value-names -o $@ -S -emit-llvm

%-raw.ll: %-unopt.ll
	opt-9 $^ -load=/workspaces/Enzyme/enzyme/build-9-Debug/Enzyme/LLVMEnzyme-9.so -enzyme -mem2reg -instsimplify -simplifycfg -o $@ -S

%-opt.ll: %-raw.ll
	opt-9 $^ -o $@ -S

$(progs): %.o: %-opt.ll
	clang-9 -O2 $^ -o $@ -lopenblas -lm
