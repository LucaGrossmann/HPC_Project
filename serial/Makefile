# Part 1 — Serial C++ baseline. Cluster build: GCC + Spack OpenBLAS.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SPACK_OPENBLAS := $(shell spack location -i openblas)
CXXFLAGS       += -I$(SPACK_OPENBLAS)/include
LDFLAGS        := -L$(SPACK_OPENBLAS)/lib -Wl,-rpath,$(SPACK_OPENBLAS)/lib \
                  -lopenblas -llapack -lm

GEN_FIXTURE := ../cuda/gen_fixture.py

.PHONY: all test clean small medium

all: tsvdm_serial

tsvdm_serial: tsvdm_serial.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Small fixture (committed): 8x8x4.
fixtures/small.bin:
	python $(GEN_FIXTURE) \
	    --rows 8 --cols 8 --slices 4 --seed 0 --out $@

# Medium fixture for a quick sanity timing: 64x64x32.
fixtures/medium.bin:
	python $(GEN_FIXTURE) \
	    --rows 64 --cols 64 --slices 32 --seed 0 --out $@

small: tsvdm_serial fixtures/small.bin
	./tsvdm_serial fixtures/small.bin

medium: tsvdm_serial fixtures/medium.bin
	./tsvdm_serial fixtures/medium.bin

test: small

clean:
	rm -f tsvdm_serial
