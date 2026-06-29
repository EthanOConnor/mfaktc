/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2014, 2015  Oliver Weihe (o.weihe@t-online.de)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "my_intrinsics.h"

#define NVCC_EXTERN
#include "sieve.h"
#include "timer.h"
#include "output.h"
#undef NVCC_EXTERN

#include "tf_debug.h"
#include "tf_96bit_base_math.cu"
#include "tf_96bit_helper.cu"

#undef INV_160_96
#include "tf_barrett96_div.cu"
#define INV_160_96
#include "tf_barrett96_div.cu"
#undef INV_160_96

#include "tf_barrett96_core.cu"

#include "gpusieve_helper.cu"

#define KERNEL_MIN_BLOCKS 2

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett92_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett92_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett92(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett88_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett88_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett88(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett87(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

#ifdef MFAKTC_BARRETT87_GS_BIT15_KERNEL
__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs_bit15(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                              int192 b_preinit, unsigned int *RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs_bit15(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                              int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
Specialized Barrett87 GPU-sieve kernel for bit levels where bit_max64 is 15.
The host dispatch in tf_common_gs.cu only launches this kernel for that exact
bit level, preserving the generic kernel for all other Barrett87 assignments.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[];

    (void)bit_max64;

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount);

    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        k_delta = k_deltas[i];

        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett87_impl<15>(f, b_preinit, initial_shifter_value, RES, 15
#ifdef DEBUG_GPU_MATH
                                     ,
                                     modbasecase_debug
#endif
        );
    }
}
#endif

#ifdef MFAKTC_BARRETT87_GS_FIXED_SHIFTER_KERNEL
#ifdef MFAKTC_BARRETT87_GS_FIXED_PROCESS_BITS
#define MFAKTC_FIXED_PROCESS_PARAM
#else
#define MFAKTC_FIXED_PROCESS_PARAM , unsigned int bits_to_process
#endif
#ifdef MFAKTC_BARRETT87_GS_FIXED_BPREINIT
#define MFAKTC_FIXED_BPREINIT_PARAM
#else
#define MFAKTC_FIXED_BPREINIT_PARAM , int192 b_preinit
#endif
#ifdef DEBUG_GPU_MATH
#define MFAKTC_FIXED_DEBUG_PARAM , unsigned int *modbasecase_debug
#define MFAKTC_FIXED_DEBUG_ARG , modbasecase_debug
#else
#define MFAKTC_FIXED_DEBUG_PARAM
#define MFAKTC_FIXED_DEBUG_ARG
#endif
#ifdef MFAKTC_BARRETT87_GS_FIXED_MIN_BLOCKS
#define MFAKTC_FIXED_KERNEL_MIN_BLOCKS MFAKTC_BARRETT87_GS_FIXED_MIN_BLOCKS
#else
#define MFAKTC_FIXED_KERNEL_MIN_BLOCKS KERNEL_MIN_BLOCKS
#endif
#ifdef MFAKTC_BARRETT87_GS_FIXED_DELTA64
#define MFAKTC_FIXED_F_DELTA64_LO                                                                                   \
    ((unsigned int)(((unsigned long long)(2U * NUM_CLASSES) *                                                       \
                     (unsigned long long)MFAKTC_BARRETT87_GS_FIXED_EXP) &                                            \
                    0xffffffffULL))
#define MFAKTC_FIXED_F_DELTA64_HI                                                                                   \
    ((unsigned int)(((unsigned long long)(2U * NUM_CLASSES) *                                                       \
                     (unsigned long long)MFAKTC_BARRETT87_GS_FIXED_EXP) >>                                           \
                    32))
__device__ static __forceinline__ void create_fixed_f_from_delta64(int96 *f, int96 f_base, unsigned int k_delta)
{
    unsigned int delta0 = __umul32(k_delta, MFAKTC_FIXED_F_DELTA64_LO);
    unsigned int delta1 = __umad32(k_delta, MFAKTC_FIXED_F_DELTA64_HI, __umul32hi(k_delta, MFAKTC_FIXED_F_DELTA64_LO));

    f->d0 = __add_cc(f_base.d0, delta0);
    f->d1 = __addc_cc(f_base.d1, delta1);
    f->d2 = __addc(f_base.d2, 0);
}
#define MFAKTC_FIXED_CREATE_F(F, F_BASE, K_DELTA, EXP) create_fixed_f_from_delta64(&(F), (F_BASE), (unsigned int)(K_DELTA))
#else
#define MFAKTC_FIXED_CREATE_F(F, F_BASE, K_DELTA, EXP)                                                               \
    do {                                                                                                             \
        (F).d0 = __add_cc((F_BASE).d0, __umul32(2 * (K_DELTA) * NUM_CLASSES, (EXP)));                                \
        (F).d1 = __addc_cc((F_BASE).d1, __umul32hi(2 * (K_DELTA) * NUM_CLASSES, (EXP)));                             \
        (F).d2 = __addc((F_BASE).d2, 0);                                                                             \
    } while (0)
#endif

__global__ void
__launch_bounds__(THREADS_PER_BLOCK, MFAKTC_FIXED_KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs_fixed_shifter(int96 k_base, unsigned int *bit_array
                                      MFAKTC_FIXED_PROCESS_PARAM
                                      MFAKTC_FIXED_BPREINIT_PARAM,
                                      unsigned int *RES
                                      MFAKTC_FIXED_DEBUG_PARAM)
/*
Specialized Barrett87 GPU-sieve kernel for one fixed exponent and bit class.
The host dispatch in tf_common_gs.cu only launches this kernel when runtime
exponent and bit class match the compile-time constants below.
*/
{
    int96 f, f_base;
    int i, total_bit_count, k_delta;
    const unsigned int fixed_exp = MFAKTC_BARRETT87_GS_FIXED_EXP;
    extern __shared__ unsigned short k_deltas[];

#ifdef MFAKTC_BARRETT87_GS_FIXED_BPREINIT
    int192 fixed_b_preinit;
#define MFAKTC_FIXED_BPREINIT_WORD_VALUE(WORD)                                                                        \
    ((MFAKTC_BARRETT87_GS_FIXED_BPREINIT_WORD == (WORD)) ?                                                            \
         (unsigned int)(MFAKTC_BARRETT87_GS_FIXED_BPREINIT_VALUE) :                                                    \
         0U)
    fixed_b_preinit.d0 = MFAKTC_FIXED_BPREINIT_WORD_VALUE(0);
    fixed_b_preinit.d1 = MFAKTC_FIXED_BPREINIT_WORD_VALUE(1);
    fixed_b_preinit.d2 = MFAKTC_FIXED_BPREINIT_WORD_VALUE(2);
    fixed_b_preinit.d3 = MFAKTC_FIXED_BPREINIT_WORD_VALUE(3);
    fixed_b_preinit.d4 = MFAKTC_FIXED_BPREINIT_WORD_VALUE(4);
    fixed_b_preinit.d5 = MFAKTC_FIXED_BPREINIT_WORD_VALUE(5);
#undef MFAKTC_FIXED_BPREINIT_WORD_VALUE
#endif

#ifdef MFAKTC_BARRETT87_GS_FIXED_PROCESS_BITS
    create_k_deltas_fixed_process(bit_array, &total_bit_count, k_deltas);
    create_fbase96_fixed_process(&f_base, k_base, fixed_exp);
#else
    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, fixed_exp, bits_to_process);
#endif

#ifdef MFAKTC_BARRETT87_GS_FIXED_BPREINIT
#define MFAKTC_FIXED_TEST_ONE(F)                                                                                       \
    test_FC96_barrett87_fixed_shifter<MFAKTC_BARRETT87_GS_FIXED_BIT_MAX64, MFAKTC_BARRETT87_GS_FIXED_SHIFTER>(         \
        (F), fixed_b_preinit, RES                                                                                       \
            MFAKTC_FIXED_DEBUG_ARG)
#else
#define MFAKTC_FIXED_TEST_ONE(F)                                                                                       \
    test_FC96_barrett87_fixed_shifter<MFAKTC_BARRETT87_GS_FIXED_BIT_MAX64, MFAKTC_BARRETT87_GS_FIXED_SHIFTER>(         \
        (F), b_preinit, RES                                                                                             \
            MFAKTC_FIXED_DEBUG_ARG)
#endif

#ifdef MFAKTC_BARRETT87_GS_FIXED_DUAL_CANDIDATE
#ifdef MFAKTC_BARRETT87_GS_FIXED_BPREINIT
#define MFAKTC_FIXED_TEST_PAIR(F0, F1)                                                                                 \
    test_FC96_barrett87_fixed_shifter_dual<MFAKTC_BARRETT87_GS_FIXED_BIT_MAX64, MFAKTC_BARRETT87_GS_FIXED_SHIFTER>(    \
        (F0), (F1), fixed_b_preinit, RES                                                                                \
            MFAKTC_FIXED_DEBUG_ARG)
#else
#define MFAKTC_FIXED_TEST_PAIR(F0, F1)                                                                                 \
    test_FC96_barrett87_fixed_shifter_dual<MFAKTC_BARRETT87_GS_FIXED_BIT_MAX64, MFAKTC_BARRETT87_GS_FIXED_SHIFTER>(    \
        (F0), (F1), b_preinit, RES                                                                                      \
            MFAKTC_FIXED_DEBUG_ARG)
#endif
#else
#define MFAKTC_FIXED_TEST_PAIR(F0, F1)                                                                                 \
    do {                                                                                                               \
        MFAKTC_FIXED_TEST_ONE(F0);                                                                                     \
        MFAKTC_FIXED_TEST_ONE(F1);                                                                                     \
    } while (0)
#endif

    for (i = threadIdx.x;
#ifdef MFAKTC_BARRETT87_GS_FIXED_DUAL_CANDIDATE
         i + THREADS_PER_BLOCK < total_bit_count;
#else
         i < total_bit_count;
#endif
         i +=
#ifdef MFAKTC_BARRETT87_GS_FIXED_DUAL_CANDIDATE
         2 *
#endif
         THREADS_PER_BLOCK) {
        k_delta = k_deltas[i];
        MFAKTC_FIXED_CREATE_F(f, f_base, k_delta, fixed_exp);

#ifdef MFAKTC_BARRETT87_GS_FIXED_DUAL_CANDIDATE
        int96 f_next;
        k_delta = k_deltas[i + THREADS_PER_BLOCK];
        MFAKTC_FIXED_CREATE_F(f_next, f_base, k_delta, fixed_exp);

        MFAKTC_FIXED_TEST_PAIR(f, f_next);
#else
        MFAKTC_FIXED_TEST_ONE(f);
#endif
    }

#ifdef MFAKTC_BARRETT87_GS_FIXED_DUAL_CANDIDATE
    for (; i < total_bit_count; i += THREADS_PER_BLOCK) {
        k_delta = k_deltas[i];
        MFAKTC_FIXED_CREATE_F(f, f_base, k_delta, fixed_exp);

        MFAKTC_FIXED_TEST_ONE(f);
    }
#endif
}
#undef MFAKTC_FIXED_TEST_PAIR
#undef MFAKTC_FIXED_TEST_ONE
#undef MFAKTC_FIXED_CREATE_F
#ifdef MFAKTC_BARRETT87_GS_FIXED_DELTA64
#undef MFAKTC_FIXED_F_DELTA64_HI
#undef MFAKTC_FIXED_F_DELTA64_LO
#endif
#undef MFAKTC_FIXED_DEBUG_PARAM
#undef MFAKTC_FIXED_DEBUG_ARG
#undef MFAKTC_FIXED_KERNEL_MIN_BLOCKS
#undef MFAKTC_FIXED_BPREINIT_PARAM
#undef MFAKTC_FIXED_PROCESS_PARAM
#endif

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett79_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett79_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett79(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett77_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett77_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett77(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett76_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett76_gs(unsigned int exp, int96 k_base, unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                        int192 b_preinit, unsigned int *RES, int bit_max64, unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += THREADS_PER_BLOCK) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett76(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

#define TF_BARRETT

#define TF_BARRETT_92BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_92BIT_GS

#define TF_BARRETT_88BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_88BIT_GS

#define TF_BARRETT_87BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_87BIT_GS

#define TF_BARRETT_79BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_79BIT_GS

#define TF_BARRETT_77BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_77BIT_GS

#define TF_BARRETT_76BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_76BIT_GS

#undef TF_BARRETT
