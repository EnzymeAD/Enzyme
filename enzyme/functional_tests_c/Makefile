
.PHONY:	all clean

OBJ := $(wildcard *.c)

#ENZYME_PLUGIN := ./../build/Enzyme/LLVMEnzyme-7.so
#export PATH = "$(PATH):$(LLVM_DIR)"

ifeq ($(CLANG_BIN_PATH),)
CLANG_BIN_PATH=${CLANG_BIN_PATH}
endif

ifeq ($(ENZYME_PLUGIN),)
ENZYME_PLUGIN=${ENZYME_PLUGIN}
endif

all: $(patsubst %.c,build/%-enzyme0,$(OBJ)) $(patsubst %.c,build/%-enzyme1,$(OBJ)) $(patsubst %.c,build/%-enzyme2,$(OBJ)) $(patsubst %.c,build/%-enzyme3,$(OBJ))

#all: $(patsubst %.c,build/%-enzyme1,$(OBJ)) $(patsubst %.c,build/%-enzyme2,$(OBJ)) $(patsubst %.c,build/%-enzyme3,$(OBJ))
#clean:
#	rm -f main main-* main.ll
#	rm -f compilercrash compilercrash-* compilercrash.ll
#	rm -f segfault segfault-* segfault.ll
#	rm -f silent_failure silent_failure-* silent_failure.ll

#ENZYME_PLUGIN = $(ENZYME_PLUGIN)

EXTRA_FLAGS = -indvars -loop-simplify -loop-rotate

# NOTE(TFK): Optimization level 0 is broken right now.
build/%-enzyme0: %.c
	@./setup.sh $(CLANG_BIN_PATH)/clang++ -O1 $(patsubst %.c,%,$<).c -S -emit-llvm -o $@.ll
	@./setup.sh $(CLANG_BIN_PATH)/opt $@.ll $(EXTRA_FLAGS) -load=$(ENZYME_PLUGIN) -enzyme -o $@.bc
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -S -emit-llvm -o $@-final.ll
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -o $@

build/%-enzyme1: %.c
	@./setup.sh $(CLANG_BIN_PATH)/clang++ -O1 $(patsubst %.c,%,$<).c -S -emit-llvm -o $@.ll
	@./setup.sh $(CLANG_BIN_PATH)/opt $@.ll $(EXTRA_FLAGS) -load=$(ENZYME_PLUGIN) -enzyme -o $@.bc
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -S -emit-llvm -o $@-final.ll
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -o $@

build/%-enzyme2: %.c
	@./setup.sh $(CLANG_BIN_PATH)/clang++ -O2 $(patsubst %.c,%,$<).c -S -emit-llvm -o $@.ll
	@./setup.sh $(CLANG_BIN_PATH)/opt $@.ll $(EXTRA_FLAGS) -load=$(ENZYME_PLUGIN) -enzyme -o $@.bc
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -S -emit-llvm -o $@-final.ll
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -o $@

build/%-enzyme3: %.c
	@./setup.sh $(CLANG_BIN_PATH)/clang++ -O3 $(patsubst %.c,%,$<).c -S -emit-llvm -o $@.ll
	@./setup.sh $(CLANG_BIN_PATH)/opt $@.ll $(EXTRA_FLAGS) -load=$(ENZYME_PLUGIN) -enzyme -o $@.bc
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -S -emit-llvm -o $@-final.ll
	@./setup.sh $(CLANG_BIN_PATH)/clang++ $@.bc -o $@


%-enzyme-test0: build/%-enzyme0
	@./$< 2> /dev/null 1> /dev/null && echo "success" || echo "FAILURE"

%-enzyme-test1: build/%-enzyme1
	@./$< 2> /dev/null 1> /dev/null && echo "success" || echo "FAILURE"

%-enzyme-test2: build/%-enzyme2
	@./$< 2> /dev/null 1> /dev/null && echo "success" || echo "FAILURE"

%-enzyme-test3: build/%-enzyme3
	@./$< 2> /dev/null 1> /dev/null && echo "success" || echo "FAILURE"

test: $(patsubst %.c,%-enzyme-test0,$(OBJ)) $(patsubst %.c,%-enzyme-test1,$(OBJ)) $(patsubst %.c,%-enzyme-test2,$(OBJ)) $(patsubst %.c,%-enzyme-test3,$(OBJ))
#test: $(patsubst %.c,%-enzyme-test1,$(OBJ)) $(patsubst %.c,%-enzyme-test2,$(OBJ)) $(patsubst %.c,%-enzyme-test3,$(OBJ))

generate-testfiles:
	python gentests.py $(patsubst %.c,%-enzyme0,$(OBJ))
	python gentests.py $(patsubst %.c,%-enzyme1,$(OBJ))
	python gentests.py $(patsubst %.c,%-enzyme2,$(OBJ))
	python gentests.py $(patsubst %.c,%-enzyme3,$(OBJ))

clean-%:
	rm -f build/%*

clean:
	rm -f build/*


