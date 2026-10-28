# engram_server (x86host, x86_64) + engram_check (any host; the Spark-side standalone exactness client).
#   make VERBS_INC=$HOME/dsv41-offload/o1/sysroot/usr/include OUT=$HOME/dsv41-offload/o1/bin
CXX ?= g++
VERBS_INC ?=
LIBIBVERBS ?= $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/aarch64-linux-gnu/libibverbs.so.1) -libverbs)
INC := -I../../transport -I.. $(if $(VERBS_INC),-I$(VERBS_INC),)
CXXFLAGS ?= -O2 -g -Wall -Wextra -std=c++17 $(INC)
OUT ?= .
T := ../../transport
all: $(OUT)/engram_server $(OUT)/engram_check
$(OUT)/engram_server: engram_server.cpp ../engram_proto.cpp ../engram_proto.h $(T)/ofl.cpp $(T)/ofl.h
	$(CXX) $(CXXFLAGS) -o $@ engram_server.cpp ../engram_proto.cpp $(T)/ofl.cpp $(LIBIBVERBS) -lpthread -ldl
$(OUT)/engram_check: ../tests/engram_check.cpp ../engram_proto.cpp ../engram_proto.h $(T)/ofl.cpp $(T)/ofl.h
	$(CXX) $(CXXFLAGS) -o $@ ../tests/engram_check.cpp ../engram_proto.cpp $(T)/ofl.cpp $(LIBIBVERBS) -lpthread
.PHONY: all
