# Part 3 — MPI. Cluster build: mpicxx wraps GCC 11.5; OpenBLAS via Spack.

CXX      := mpicxx
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SPACK_OPENBLAS := $(shell spack location -i openblas)
CXXFLAGS       += -I$(SPACK_OPENBLAS)/include
LDFLAGS        := -L$(SPACK_OPENBLAS)/lib -Wl,-rpath,$(SPACK_OPENBLAS)/lib \
                  -lopenblas -llapack -lm

.PHONY: all test clean small medium

all: tsvdm_mpi

tsvdm_mpi: tsvdm_mpi.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Reuse the serial fixtures rather than regenerating them.
small: tsvdm_mpi
	mpirun -np 2 ./tsvdm_mpi ../serial/fixtures/small.bin

medium: tsvdm_mpi
	mpirun -np 4 ./tsvdm_mpi ../serial/fixtures/medium.bin

test: small

clean:
	rm -f tsvdm_mpi
