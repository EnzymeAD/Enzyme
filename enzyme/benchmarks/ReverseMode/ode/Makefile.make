# RUN: cd %S && LD_LIBRARY_PATH="%bldpath:$LD_LIBRARY_PATH" BENCH="%bench" LOAD="%loadEnzyme" LOADCLANG="%loadClangEnzyme" make -B tuned.exe VERBOSE=1 -f %s

.PHONY: clean

clean:
	rm -f *.exe *.o results.txt ode.txt

logger.exe: ode.cpp fp-logger.cpp
	clang++ $(BENCH) $(LOADCLANG) ode.cpp fp-logger.cpp -O3 -mllvm -enzyme-inline -ffast-math -fno-finite-math-only -o $@ -DLOGGING

ode.txt: logger.exe
	./$^ 1000000 | tee $@

tuned.exe: ode.cpp ode.txt
	clang++ $(BENCH) $(LOADCLANG) ode.cpp -O3 -ffast-math -fno-finite-math-only -o $@ \
		-mllvm -enzyme-inline \
		-mllvm -enzyme-enable-fpopt \
		-mllvm -fpopt-log-path=ode.txt \
		-mllvm -fpopt-enable-solver \
		-mllvm -fpopt-enable-pt \
		-mllvm -fpopt-target-func-regex=foobar \
		-mllvm -fpopt-comp-cost-budget=0 \
		-mllvm -fpopt-num-samples=1024 \
		-mllvm -fpopt-cost-dom-thres=0.0 \
		-mllvm -fpopt-acc-dom-thres=0.0 \
		-mllvm -enzyme-print-fpopt \
		-mllvm -fpopt-show-table \
		-mllvm -fpopt-cache-path=cache \
		-mllvm -herbie-timeout=1000 \
		-mllvm -herbie-num-threads=12 \
		-mllvm --fpopt-cost-model-path=cm.csv

results.txt: tuned.exe
	./$^ 1000000 | tee $@