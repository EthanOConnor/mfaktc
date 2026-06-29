# configure path to CUDA Toolkit
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE = -I$(CUDA_DIR)/include/
CUDA_LIB = -L$(CUDA_DIR)/lib64/

# compiler settings for .c files (CPU)
CC = gcc
CFLAGS = -Wall -Wextra -O3 -flto -malign-double -ffunction-sections -fdata-sections -Wl,--gc-sections $(CUDA_INCLUDE)
CFLAGS_EXTRA_SIEVE = -funroll-all-loops

# compiler settings for .cu files (GPU)
NVCC = nvcc
NVCCFLAGS = $(CUDA_INCLUDE) --ptxas-options=-v -O3 -Wno-deprecated-gpu-targets
THREADS_PER_BLOCK ?= 256
ifneq ($(THREADS_PER_BLOCK),256)
NVCCFLAGS += -DMFAKTC_THREADS_PER_BLOCK=$(THREADS_PER_BLOCK)
CFLAGS += -DMFAKTC_THREADS_PER_BLOCK=$(THREADS_PER_BLOCK)
endif
LINEINFO ?= 0
ifeq ($(LINEINFO),1)
NVCCFLAGS += -lineinfo
endif
MUL96_PTX ?= 0
ifeq ($(MUL96_PTX),1)
NVCCFLAGS += -DMFAKTC_FORCE_MUL96_PTX
endif
BARRETT87_BIT15 ?= 0
ifeq ($(BARRETT87_BIT15),1)
NVCCFLAGS += -DMFAKTC_BARRETT87_BIT15_SPECIAL
endif
BARRETT87_GS_BIT15_KERNEL ?= 0
ifeq ($(BARRETT87_GS_BIT15_KERNEL),1)
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_BIT15_KERNEL
endif
BARRETT87_GS_FIXED_EXP ?= 0
BARRETT87_GS_FIXED_BIT_MAX64 ?= 0
BARRETT87_GS_FIXED_SHIFTER ?= 0
BARRETT87_GS_FIXED_BPREINIT_WORD ?= -1
BARRETT87_GS_FIXED_BPREINIT_VALUE ?= 0
BARRETT87_GS_FIXED_INITIAL_A_SHIFT ?= -1
BARRETT87_GS_FIXED_PROCESS_BITS ?= 0
BARRETT87_GS_FIXED_SQUARE160_GUARD ?= 0
ifneq ($(BARRETT87_GS_FIXED_EXP),0)
ifeq ($(BARRETT87_GS_FIXED_BIT_MAX64),0)
$(error BARRETT87_GS_FIXED_BIT_MAX64 must be set when BARRETT87_GS_FIXED_EXP is set)
endif
ifeq ($(BARRETT87_GS_FIXED_SHIFTER),0)
$(error BARRETT87_GS_FIXED_SHIFTER must be set when BARRETT87_GS_FIXED_EXP is set)
endif
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_SHIFTER_KERNEL
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_EXP=$(BARRETT87_GS_FIXED_EXP)
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_BIT_MAX64=$(BARRETT87_GS_FIXED_BIT_MAX64)
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_SHIFTER=$(BARRETT87_GS_FIXED_SHIFTER)
ifneq ($(BARRETT87_GS_FIXED_BPREINIT_WORD),-1)
ifeq ($(BARRETT87_GS_FIXED_BPREINIT_VALUE),0)
$(error BARRETT87_GS_FIXED_BPREINIT_VALUE must be set when BARRETT87_GS_FIXED_BPREINIT_WORD is set)
endif
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_BPREINIT
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_BPREINIT_WORD=$(BARRETT87_GS_FIXED_BPREINIT_WORD)
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_BPREINIT_VALUE=$(BARRETT87_GS_FIXED_BPREINIT_VALUE)
endif
ifneq ($(BARRETT87_GS_FIXED_INITIAL_A_SHIFT),-1)
ifeq ($(BARRETT87_GS_FIXED_BPREINIT_WORD),-1)
$(error BARRETT87_GS_FIXED_BPREINIT_WORD must be set when BARRETT87_GS_FIXED_INITIAL_A_SHIFT is set)
endif
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_INITIAL_REDUCTION
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT=$(BARRETT87_GS_FIXED_INITIAL_A_SHIFT)
endif
ifneq ($(BARRETT87_GS_FIXED_PROCESS_BITS),0)
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_PROCESS_BITS=$(BARRETT87_GS_FIXED_PROCESS_BITS)
endif
ifeq ($(BARRETT87_GS_FIXED_SQUARE160_GUARD),1)
NVCCFLAGS += -DMFAKTC_BARRETT87_GS_FIXED_SQUARE160_GUARD
endif
endif
SQUARE96_U16 ?= 0
ifeq ($(SQUARE96_U16),1)
NVCCFLAGS += -DMFAKTC_FORCE_SQUARE96_U16
endif
KDELTA_SMEM_U16 ?= 0
ifeq ($(KDELTA_SMEM_U16),1)
NVCCFLAGS += -DMFAKTC_KDELTA_SMEM_U16
endif
KDELTA_WARP_SCAN ?= 0
ifeq ($(KDELTA_WARP_SCAN),1)
NVCCFLAGS += -DMFAKTC_KDELTA_WARP_SCAN
endif
GSPROCESS_MAX64 ?= 0
ifeq ($(GSPROCESS_MAX64),1)
NVCCFLAGS += -DMFAKTC_GPU_SIEVE_PROCESS_MAX64
CFLAGS += -DMFAKTC_GPU_SIEVE_PROCESS_MAX64
endif

# generate code for compute capabilities - see this table for supported
# versions: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

# older deprecated versions - uncomment to enable
#
# note: mfaktc 0.24.x dropped support for compute capability 1.x GPUs - use the '0.23' branch for older devices
# NVCCFLAGS += --generate-code arch=compute_20,code=sm_20 # Fermi GPUs
# NVCCFLAGS += --generate-code arch=compute_30,code=sm_30 # early Kepler GPUs
# NVCCFLAGS += --generate-code arch=compute_35,code=sm_35 # later Kepler GPUs; funnel shift added in 3.2

# removed in CUDA 13.x - uncomment to enable
# NVCCFLAGS += --generate-code arch=compute_50,code=sm_50 # Maxwell GPUs
# NVCCFLAGS += --generate-code arch=compute_60,code=sm_60 # Pascal GPUs
# NVCCFLAGS += --generate-code arch=compute_61,code=sm_61
# NVCCFLAGS += --generate-code arch=compute_62,code=sm_62
# NVCCFLAGS += --generate-code arch=compute_70,code=sm_70 # Volta GPUs
# NVCCFLAGS += --generate-code arch=compute_72,code=sm_72

# currently supported
NVCCFLAGS += --generate-code arch=compute_75,code=sm_75 # Turing GPUs; slightly faster than 7.0
NVCCFLAGS += --generate-code arch=compute_80,code=sm_80 # Ampere GPUs
NVCCFLAGS += --generate-code arch=compute_86,code=sm_86
NVCCFLAGS += --generate-code arch=compute_87,code=sm_87
NVCCFLAGS += --generate-code arch=compute_89,code=sm_89 # Ada Lovelace GPUs
NVCCFLAGS += --generate-code arch=compute_90,code=sm_90 # Hopper GPUs
NVCCFLAGS += --generate-code arch=compute_120,code=sm_120 # Blackwell GPUs

# pass some options to the C host compiler
NVCCFLAGS += --compiler-options=-Wall

# Linker
LD = gcc
LDFLAGS = -flto -fPIC -Wl,--gc-sections $(CUDA_LIB) -lcudart_static -lm -lstdc++

INSTALL = install

##############################################################################

CSRC  = sieve.c timer.c parse.c read_config.c mfaktc.c checkpoint.c \
	filelocking.c signal_handler.c output.c crc.c
CUSRC = tf_96bit.cu tf_barrett96.cu tf_barrett96_gs.cu gpusieve.cu \
        cuda_utils.cu

COBJS  = $(CSRC:.c=.o)
CUOBJS = $(CUSRC:.cu=.o) tf_75bit.o

##############################################################################

all: ../mfaktc ../mfaktc.ini

../mfaktc : $(COBJS) $(CUOBJS)
	$(LD) $^ -o $@ $(LDFLAGS)

clean :
	rm -f *.o *~

sieve.o : sieve.c
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA_SIEVE) -c $< -o $@

tf_75bit.o : tf_96bit.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ -DSHORTCUT_75BIT

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

../%.ini : %.ini
	$(INSTALL) -m 644 $< ..

.PHONY: all clean
