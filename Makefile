include config.mk

## Test binaries
TEST_SRC := $(wildcard test/*.cpp)
TEST_OUTPUT = build/test
TEST_BINS := $(patsubst test/%.cpp,$(TEST_OUTPUT)/%,$(TEST_SRC))

## GPU Test binaries
GPU_TEST_SRC := $(wildcard test/gpu/*.cpp)
GPU_TEST_OUTPUT = build/test/gpu
GPU_TEST_BINS := $(patsubst test/gpu/%.cpp,$(GPU_TEST_OUTPUT)/%,$(GPU_TEST_SRC))

BIN_OUTPUT = build/bin

COO_BINS=\
    $(BIN_OUTPUT)/coo_serial_O1 $(BIN_OUTPUT)/coo_omp_O1 $(BIN_OUTPUT)/coo_omp_gpu_O1 \
    $(BIN_OUTPUT)/coo_serial_O2 $(BIN_OUTPUT)/coo_omp_O2 $(BIN_OUTPUT)/coo_omp_gpu_O2 \
    $(BIN_OUTPUT)/coo_serial_O3 $(BIN_OUTPUT)/coo_omp_O3 $(BIN_OUTPUT)/coo_omp_gpu_O3

CSR_BINS=\
    $(BIN_OUTPUT)/csr_serial_O1 $(BIN_OUTPUT)/csr_omp_O1 $(BIN_OUTPUT)/csr_omp_gpu_O1 \
    $(BIN_OUTPUT)/csr_serial_O2 $(BIN_OUTPUT)/csr_omp_O2 $(BIN_OUTPUT)/csr_omp_gpu_O2 \
    $(BIN_OUTPUT)/csr_serial_O3 $(BIN_OUTPUT)/csr_omp_O3 $(BIN_OUTPUT)/csr_omp_gpu_O3

ELL_BINS=\
    $(BIN_OUTPUT)/ell_serial_O1 $(BIN_OUTPUT)/ell_omp_O1 $(BIN_OUTPUT)/ell_omp_gpu_O1 \
    $(BIN_OUTPUT)/ell_serial_O2 $(BIN_OUTPUT)/ell_omp_O2 $(BIN_OUTPUT)/ell_omp_gpu_O2 \
    $(BIN_OUTPUT)/ell_serial_O3 $(BIN_OUTPUT)/ell_omp_O3 $(BIN_OUTPUT)/ell_omp_gpu_O3

BCSR_BINS=\
    $(BIN_OUTPUT)/bcsr_serial_O1 $(BIN_OUTPUT)/bcsr_omp_O1 $(BIN_OUTPUT)/bcsr_omp_gpu_O1 \
    $(BIN_OUTPUT)/bcsr_serial_O2 $(BIN_OUTPUT)/bcsr_omp_O2 $(BIN_OUTPUT)/bcsr_omp_gpu_O2 \
    $(BIN_OUTPUT)/bcsr_serial_O3 $(BIN_OUTPUT)/bcsr_omp_O3 $(BIN_OUTPUT)/bcsr_omp_gpu_O3

BENCH_BINS=\
	$(COO_BINS) \
	$(CSR_BINS) \
	$(ELL_BINS) \
	$(BCSR_BINS)
    
CUSPARSE_BINS=\
	$(BIN_OUTPUT)/coo_cusparse \
	$(BIN_OUTPUT)/csr_cusparse

## The core
all: check_dir build/libspmm.a build/fmtbcsr $(TEST_BINS) $(GPU_TEST_BINS) $(BENCH_BINS)
cusparse: $(CUSPARSE_BINS)

.PHONY: clean_dir
check_dir:
	if [ ! -d ./build ]; then mkdir build; fi; \
	if [ ! -d ./build/test ] ; then mkdir build/test; fi; \
	if [ ! -d ./build/test/gpu ] ; then mkdir build/test/gpu; fi; \
	if [ ! -d ./$(BIN_OUTPUT) ] ; then mkdir $(BIN_OUTPUT); fi; \

##
## Build the static library
##
build/libspmm.a: build/spmm.o
	ar rvs build/libspmm.a build/spmm.o

build/spmm.o: src/spmm.cpp src/spmm.h src/csr/csr.h src/ell/ell.h src/bcsr/bcsr.h
	$(CXX) src/spmm.cpp -c -o build/spmm.o -O2 -fPIE -march=native -fopenmp

##
## Build the BCSR formatting utility
##
build/fmtbcsr: src/format_bcsr.cpp build/libspmm.a
	$(CXX) -Isrc/bcsr src/format_bcsr.cpp -o build/fmtbcsr $(CXXFLAGS)
	
##
## Build the test binaries
##
$(TEST_OUTPUT)/%: test/%.cpp build/libspmm.a
	$(CXX) $< -o $@ $(CXXFLAGS) $(OMPFLAGS)

$(GPU_TEST_OUTPUT)/%: test/gpu/%.cpp build/libspmm.a
	$(CXX) $< -o $@ $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS)
	
########################################################################
BASE=src/main.cpp build/libspmm.a
DEPS=build/libspmm.a src/main.cpp

##
## Builds the COO executables
##
# Serial
$(BIN_OUTPUT)/coo_serial_O1: src/coo/coo_serial.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_serial.cpp -o $(BIN_OUTPUT)/coo_serial_O1 $(CXXFLAGS) -O1
	
$(BIN_OUTPUT)/coo_serial_O2: src/coo/coo_serial.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_serial.cpp -o $(BIN_OUTPUT)/coo_serial_O2 $(CXXFLAGS) -O2

$(BIN_OUTPUT)/coo_serial_O3: src/coo/coo_serial.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_serial.cpp -o $(BIN_OUTPUT)/coo_serial_O3 $(CXXFLAGS) -O3

# OMP
$(BIN_OUTPUT)/coo_omp_O1: src/coo/coo_omp.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_omp.cpp -o $(BIN_OUTPUT)/coo_omp_O1 $(CXXFLAGS) $(OMPFLAGS) -O1
	
$(BIN_OUTPUT)/coo_omp_O2: src/coo/coo_omp.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_omp.cpp -o $(BIN_OUTPUT)/coo_omp_O2 $(CXXFLAGS) $(OMPFLAGS) -O2

$(BIN_OUTPUT)/coo_omp_O3: src/coo/coo_omp.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_omp.cpp -o $(BIN_OUTPUT)/coo_omp_O3 $(CXXFLAGS) $(OMPFLAGS) -O3

# OMP GPU
$(BIN_OUTPUT)/coo_omp_gpu_O1: src/coo/coo_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_omp_gpu.cpp -o $(BIN_OUTPUT)/coo_omp_gpu_O1 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O1

$(BIN_OUTPUT)/coo_omp_gpu_O2: src/coo/coo_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_omp_gpu.cpp -o $(BIN_OUTPUT)/coo_omp_gpu_O2 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O2

$(BIN_OUTPUT)/coo_omp_gpu_O3: src/coo/coo_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/coo $(BASE) src/coo/coo_omp_gpu.cpp -o $(BIN_OUTPUT)/coo_omp_gpu_O3 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O3

# CuSparse COO
$(BIN_OUTPUT)/coo_cusparse: src/coo/coo_cusparse.cpp $(DEPS)
	nvcc -Isrc/coo $(BASE) src/coo/coo_cusparse.cpp -o $(BIN_OUTPUT)/coo_cusparse -Isrc build/libspmm.a -lcusparse -lgomp

##################################################################################################################################

##
## Builds the CSR executables
##
# Serial
$(BIN_OUTPUT)/csr_serial_O1: src/csr/csr_serial.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_serial.cpp -o $(BIN_OUTPUT)/csr_serial_O1 $(CXXFLAGS) -O1

$(BIN_OUTPUT)/csr_serial_O2: src/csr/csr_serial.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_serial.cpp -o $(BIN_OUTPUT)/csr_serial_O2 $(CXXFLAGS) -O2
	
$(BIN_OUTPUT)/csr_serial_O3: src/csr/csr_serial.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_serial.cpp -o $(BIN_OUTPUT)/csr_serial_O3 $(CXXFLAGS) -O3

# OMP
$(BIN_OUTPUT)/csr_omp_O1: src/csr/csr_omp.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_omp.cpp -o $(BIN_OUTPUT)/csr_omp_O1 $(CXXFLAGS) $(OMPFLAGS) -O1

$(BIN_OUTPUT)/csr_omp_O2: src/csr/csr_omp.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_omp.cpp -o $(BIN_OUTPUT)/csr_omp_O2 $(CXXFLAGS) $(OMPFLAGS) -O2

$(BIN_OUTPUT)/csr_omp_O3: src/csr/csr_omp.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_omp.cpp -o $(BIN_OUTPUT)/csr_omp_O3 $(CXXFLAGS) $(OMPFLAGS) -O3

# OMP GPU
$(BIN_OUTPUT)/csr_omp_gpu_O1: src/csr/csr_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_omp_gpu.cpp -o $(BIN_OUTPUT)/csr_omp_gpu_O1 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O1

$(BIN_OUTPUT)/csr_omp_gpu_O2: src/csr/csr_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_omp_gpu.cpp -o $(BIN_OUTPUT)/csr_omp_gpu_O2 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O2

$(BIN_OUTPUT)/csr_omp_gpu_O3: src/csr/csr_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/csr $(BASE) src/csr/csr_omp_gpu.cpp -o $(BIN_OUTPUT)/csr_omp_gpu_O3 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O3

# CuSparse CSR
$(BIN_OUTPUT)/csr_cusparse: src/csr/csr_cusparse.cpp $(DEPS)
	nvcc -Isrc/csr $(BASE) src/csr/csr_cusparse.cpp -o $(BIN_OUTPUT)/csr_cusparse -Isrc build/libspmm.a -lcusparse -lgomp
	
##################################################################################################################################

##
## Builds the ELL executables
##
# Serial
$(BIN_OUTPUT)/ell_serial_O1: src/ell/ell_serial.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_serial.cpp -o $(BIN_OUTPUT)/ell_serial_O1 $(CXXFLAGS) -O1

$(BIN_OUTPUT)/ell_serial_O2: src/ell/ell_serial.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_serial.cpp -o $(BIN_OUTPUT)/ell_serial_O2 $(CXXFLAGS) -O2

$(BIN_OUTPUT)/ell_serial_O3: src/ell/ell_serial.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_serial.cpp -o $(BIN_OUTPUT)/ell_serial_O3 $(CXXFLAGS) -O3

# OMP Parallel
$(BIN_OUTPUT)/ell_omp_O1: src/ell/ell_omp.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_omp.cpp -o $(BIN_OUTPUT)/ell_omp_O1 $(CXXFLAGS) $(OMPFLAGS) -O1

$(BIN_OUTPUT)/ell_omp_O2: src/ell/ell_omp.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_omp.cpp -o $(BIN_OUTPUT)/ell_omp_O2 $(CXXFLAGS) $(OMPFLAGS) -O2

$(BIN_OUTPUT)/ell_omp_O3: src/ell/ell_omp.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_omp.cpp -o $(BIN_OUTPUT)/ell_omp_O3 $(CXXFLAGS) $(OMPFLAGS) -O3

# OMP GPU
$(BIN_OUTPUT)/ell_omp_gpu_O1: src/ell/ell_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_omp_gpu.cpp -o $(BIN_OUTPUT)/ell_omp_gpu_O1 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O1

$(BIN_OUTPUT)/ell_omp_gpu_O2: src/ell/ell_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_omp_gpu.cpp -o $(BIN_OUTPUT)/ell_omp_gpu_O2 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O2

$(BIN_OUTPUT)/ell_omp_gpu_O3: src/ell/ell_omp_gpu.cpp $(DEPS)
	$(CXX) -Isrc/ell $(BASE) src/ell/ell_omp_gpu.cpp -o $(BIN_OUTPUT)/ell_omp_gpu_O3 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O3

##################################################################################################################################
	
##
## Builds the BCSR executables
##
# Serial
$(BIN_OUTPUT)/bcsr_serial_O1: src/bcsr/bcsr_serial.cpp $(DEPS)
	$(CXX) -Isrc/bcsr $(BASE) src/bcsr/bcsr_serial.cpp -o $(BIN_OUTPUT)/bcsr_serial_O1 $(CXXFLAGS) -O1

$(BIN_OUTPUT)/bcsr_serial_O2: src/bcsr/bcsr_serial.cpp $(DEPS)
	$(CXX) -Isrc/bcsr $(BASE) src/bcsr/bcsr_serial.cpp -o $(BIN_OUTPUT)/bcsr_serial_O2 $(CXXFLAGS) -O2

$(BIN_OUTPUT)/bcsr_serial_O3: src/bcsr/bcsr_serial.cpp $(DEPS)
	$(CXX) -Isrc/bcsr $(BASE) src/bcsr/bcsr_serial.cpp -o $(BIN_OUTPUT)/bcsr_serial_O3 $(CXXFLAGS) -O3

# OMP
$(BIN_OUTPUT)/bcsr_omp_O1: src/bcsr/bcsr_omp.cpp $(DEPS)
	$(CXX)  -Isrc/bcsr $(BASE) src/bcsr/bcsr_omp.cpp -o $(BIN_OUTPUT)/bcsr_omp_O1 $(CXXFLAGS) $(OMPFLAGS) -O1

$(BIN_OUTPUT)/bcsr_omp_O2: src/bcsr/bcsr_omp.cpp $(DEPS)
	$(CXX)  -Isrc/bcsr $(BASE) src/bcsr/bcsr_omp.cpp -o $(BIN_OUTPUT)/bcsr_omp_O2 $(CXXFLAGS) $(OMPFLAGS) -O2

$(BIN_OUTPUT)/bcsr_omp_O3: src/bcsr/bcsr_omp.cpp $(DEPS)
	$(CXX)  -Isrc/bcsr $(BASE) src/bcsr/bcsr_omp.cpp -o $(BIN_OUTPUT)/bcsr_omp_O3 $(CXXFLAGS) $(OMPFLAGS) -O3

# OMP GPU
$(BIN_OUTPUT)/bcsr_omp_gpu_O1: src/bcsr/bcsr_omp_gpu.cpp $(DEPS)
	$(CXX)  -Isrc/bcsr $(BASE) src/bcsr/bcsr_omp_gpu.cpp -o $(BIN_OUTPUT)/bcsr_omp_gpu_O1 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O1

$(BIN_OUTPUT)/bcsr_omp_gpu_O2: src/bcsr/bcsr_omp_gpu.cpp $(DEPS)
	$(CXX)  -Isrc/bcsr $(BASE) src/bcsr/bcsr_omp_gpu.cpp -o $(BIN_OUTPUT)/bcsr_omp_gpu_O2 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O2

$(BIN_OUTPUT)/bcsr_omp_gpu_O3: src/bcsr/bcsr_omp_gpu.cpp $(DEPS)
	$(CXX)  -Isrc/bcsr $(BASE) src/bcsr/bcsr_omp_gpu.cpp -o $(BIN_OUTPUT)/bcsr_omp_gpu_O3 $(CXXFLAGS) $(OMPFLAGS) $(GPUFLAGS) -O3

########################################################################

##
## Clean target
##
.PHONY: clean
clean:
	rm -rf build/*


