# Part 1 — Serial C++ baseline. Cluster build: GCC + Spack OpenBLAS.

CXX      := g++
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wextra

SPACK_OPENBLAS := $(shell spack location -i openblas)
CXXFLAGS       += -I$(SPACK_OPENBLAS)/include
LDFLAGS        := -L$(SPACK_OPENBLAS)/lib -Wl,-rpath,$(SPACK_OPENBLAS)/lib \
                  -lopenblas -lm

.PHONY: all clean

all: tsvdm_serial

tsvdm_serial: tsvdm_serial.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f tsvdm_serial
