# ofl transport: libofl.so + tools. Builds on x86host (x86_64; verbs headers from the extracted libibverbs-dev 50.0 deb,
# VERBS_INC) and on a DGX Spark host or in the serve container (aarch64; system headers). No install step.
#   make                      # libofl.so ofl_pingpong ofl_probe
#   make VERBS_INC=$HOME/dsv41-offload/o1/sysroot/usr/include
CXX ?= g++
CC ?= gcc
VERBS_INC ?=
LIBIBVERBS ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/aarch64-linux-gnu/libibverbs.so.1) -libverbs)
INC := $(if $(VERBS_INC),-I$(VERBS_INC),)
CXXFLAGS ?= -O2 -g -Wall -Wextra -std=c++17 -fPIC $(INC)
CFLAGS ?= -O2 -g -Wall -std=c11 $(INC)
OUT ?= .

all: $(OUT)/libofl.so $(OUT)/ofl_pingpong $(OUT)/ofl_probe

$(OUT)/libofl.so: ofl.cpp ofl.h
	$(CXX) $(CXXFLAGS) -shared -o $@ ofl.cpp $(LIBIBVERBS) -lpthread

$(OUT)/ofl_pingpong: ofl_pingpong.cpp ofl.cpp ofl.h
	$(CXX) $(CXXFLAGS) -o $@ ofl_pingpong.cpp ofl.cpp $(LIBIBVERBS) -lpthread

$(OUT)/ofl_probe: ofl_probe.c
	$(CC) $(CFLAGS) -o $@ ofl_probe.c $(LIBIBVERBS)

clean:
	rm -f $(OUT)/libofl.so $(OUT)/ofl_pingpong $(OUT)/ofl_probe

.PHONY: all clean
