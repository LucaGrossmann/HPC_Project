# Part 3 — MPI. Cluster build: mpicxx wraps GCC 11.5; OpenBLAS via Spack.

CXX      := mpicxx
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wextra

SPACK_OPENBLAS := $(shell spack location -i openblas)
CXXFLAGS       += -I$(SPACK_OPENBLAS)/include
LDFLAGS        := -L$(SPACK_OPENBLAS)/lib -Wl,-rpath,$(SPACK_OPENBLAS)/lib \
                  -lopenblas -lm

.PHONY: all clean

all: tsvdm_mpi

tsvdm_mpi: tsvdm_mpi.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f tsvdm_mpi
