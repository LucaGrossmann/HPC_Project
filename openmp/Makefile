# Part 2 — OpenMP. Cluster build: GCC + -fopenmp + Spack OpenBLAS.
# Mirrors serial/Makefile; only difference is -fopenmp.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SPACK_OPENBLAS := $(shell spack location -i openblas)
CXXFLAGS       += -fopenmp -I$(SPACK_OPENBLAS)/include
LDFLAGS        := -fopenmp -L$(SPACK_OPENBLAS)/lib -Wl,-rpath,$(SPACK_OPENBLAS)/lib \
                  -lopenblas -llapack -lm

GEN_FIXTURE := ../cuda/gen_fixture.py

.PHONY: all test clean small medium

all: tsvdm_openmp

tsvdm_openmp: tsvdm_openmp.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Reuse the serial fixtures rather than regenerating them.
small: tsvdm_openmp
	./tsvdm_openmp ../serial/fixtures/small.bin

medium: tsvdm_openmp
	./tsvdm_openmp ../serial/fixtures/medium.bin

test: small

clean:
	rm -f tsvdm_openmp
