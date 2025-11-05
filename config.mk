CC=clang
CXX=clang++

#CC=/data/opt/llvm/bin/clang
#CXX=/data/opt/llvm/bin/clang++

CXXFLAGS=-Isrc build/libspmm.a -std=c++17 -march=native -fopenmp
#OMPFLAGS=-fopenmp
#GPUFLAGS=-fopenmp-targets=nvptx64

