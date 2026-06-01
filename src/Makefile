ACPP_FLAGS := -O3 --acpp-targets="generic"
NVCC_FLAGS := -O3

# Allow template parameters to be set via environment or command line
BM ?= 32
BN ?= 32
BK ?= 32
TM ?= 1
TN ?= 1

# Defaults used only by `make stencil` / `make all`.
STENCIL_BM ?= 16
STENCIL_BN ?= 64
STENCIL_BK ?= 0
STENCIL_TM ?= 1
STENCIL_TN ?= 1

# Append template parameter defines to compiler flags
ACPP_FLAGS += -D_BM=$(BM) -D_BN=$(BN) -D_BK=$(BK) -D_TM=$(TM) -D_TN=$(TN)
NVCC_FLAGS += -D_BM=$(BM) -D_BN=$(BN) -D_BK=$(BK) -D_TM=$(TM) -D_TN=$(TN)

.PHONY: all matmul stencil clean

# Default libraries built by `make all`
all: matmul stencil

matmul: kernel_matmul_sycl.so kernel_matmul_cuda.so

stencil:
	$(MAKE) kernel_stencil_sycl.so BM=$(STENCIL_BM) BN=$(STENCIL_BN) BK=$(STENCIL_BK) TM=$(STENCIL_TM) TN=$(STENCIL_TN)
	$(MAKE) kernel_stencil_cuda.so BM=$(STENCIL_BM) BN=$(STENCIL_BN) BK=$(STENCIL_BK) TM=$(STENCIL_TM) TN=$(STENCIL_TN)

kernel_%_sycl.so: kernel_%_sycl.o
	acpp $(ACPP_FLAGS) -shared -o $@ $^

kernel_%_sycl.o: kernel_%_sycl.cpp
	acpp $(ACPP_FLAGS) -fPIC -c $< -o $@

kernel_%_cuda.so: kernel_%_cuda.o
	nvcc $(NVCC_FLAGS) -shared -Xcompiler -fPIC -o $@ $^

kernel_%_cuda.o: kernel_%_cuda.cu
	nvcc $(NVCC_FLAGS) -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f kernel_*_sycl.o kernel_*_sycl.so kernel_*_cuda.o kernel_*_cuda.so
