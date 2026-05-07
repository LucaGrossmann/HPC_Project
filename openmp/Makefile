# Part 2 — OpenMP. Cluster build: GCC + -fopenmp + Spack OpenBLAS.
# Mirrors serial/Makefile; only difference is -fopenmp.

CXX      := g++
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wextra

# Keep debug-info + frame pointer so VTune can attribute samples accurately.
CXXFLAGS += -g -fno-omit-frame-pointer

SPACK_OPENBLAS := $(shell spack location -i openblas)
CXXFLAGS       += -fopenmp -I$(SPACK_OPENBLAS)/include
LDFLAGS        := -fopenmp -L$(SPACK_OPENBLAS)/lib -Wl,-rpath,$(SPACK_OPENBLAS)/lib \
                  -lopenblas -lm

.PHONY: all clean

all: tsvdm_openmp

tsvdm_openmp: tsvdm_openmp.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f tsvdm_openmp
