/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2013, 2015  Oliver Weihe (o.weihe@t-online.de)

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

/*
This file contains the core function for the barrett based kernels. Each
function handles exactly on factor candidate. Those functions are called
from the kernels with CPU sieving (tf_barrett96.cu) or the kernels with GPU
sieving (tf_barrett96_gs.cu). The only difference is that the GPU kernels
use a preshifted value for "shifter" while the CPU sieve kernels shift the
"shifter" inplace. For some reason the GPU sieve kernels run slower when the
shift is done inplace and the CPU sieve kernels are slower when the shift is
precomputed... This behaviour is controlled by the define CPU_SIEVE.
*/

__device__ static void test_FC96_barrett92(int96 f, int192 b, unsigned int shifter, unsigned int *RES, int bit_max64
#ifdef CPU_SIEVE
                                           ,
                                           int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           ,
                                           unsigned int *modbasecase_debug
#endif
)
{
    int96 a, u;
    int192 tmp192;
    int96 tmp96;
    float ff;

    trace_96_textmsg(__FILE__, __LINE__, f, "--- barrett92 start ---");
    // ff = f as float, needed in mod_192_96().
    // Precalculated here since it is the same for all steps in the following loop
    ff = __uint2float_rn(f.d2);
    ff = ff * 4294967296.0f + __uint2float_rn(f.d1); /* f.d0 ignored because lower limit for this kernel are 64 bit
                                                        which yields at least 32 significant digits without f.d0! */
    ff = __int_as_float(0x3f7ffffb) / ff; // just a little bit below 1.0f so we always underestimate the quotient

    tmp192.d5 = 1 << (bit_max64 - 1); // tmp192 = 2^(95 + bits_in_f)
    tmp192.d4 = 0;
    tmp192.d3 = 0;
    tmp192.d2 = 0;
    tmp192.d1 = 0;
    tmp192.d0 = 0;

#ifndef DEBUG_GPU_MATH
    div_192_96(&u, tmp192, f, ff); // u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#else
    div_192_96(&u, tmp192, f, ff, modbasecase_debug); // u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#endif
    trace_96_96(__FILE__, __LINE__, f, "u", u);

    a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1); // a = floor(b / 2 ^ (bits_in_f - 1))
    a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);
    a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96_192_no_low3(&tmp192, a, u); /* tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)
                                          (ignore the floor functions for now) */

    a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

    // clang-format off
    tmp96.d0 = __sub_cc(b.d0, tmp96.d0);  // Compute the remainder
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1); // we do not need the upper digits of b and tmp96 because
    tmp96.d2 = __subc(b.d2, tmp96.d2);    // the result is 0 after subtraction!
    trace_96_96(__FILE__, __LINE__, f, "a", a);
    // clang-format on

#ifdef CPU_SIEVE
    shifter <<= 32 - shiftcount;
#endif
    while (shifter) {
        trace_96_textmsg(__FILE__, __LINE__, f, "--- main loop start ---");
#ifndef DEBUG_GPU_MATH
        mod_simple_96(&a, tmp96, f, ff); /* Adjustment. The code above/below may produce an a
                                            that is too large by up to 11 times f. */
#else
        mod_simple_96(&a, tmp96, f, ff, bit_max64 - 1, bit_max64, 11,
                      modbasecase_debug); // bit_max - 1 = bit_min (this kernel handles only single bit levels)
#endif
        // Since mod_simple_96 does not do a complete adjustment we need to allow one bit
        // for that.  Thus, at this point a can be 93 bits.

        // On input a is at most 93 bits (see mod_simple_96 above)
        square_96_192(&b, a); // b = a^2, b is at most 186 bits
        trace_96_192(__FILE__, __LINE__, f, "b", b);

        a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95 bits
        a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);
        a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);
        trace_96_96(__FILE__, __LINE__, f, "a", a);

        mul_96_192_no_low3(&tmp192, a, u); /* tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)
                                              (ignore the floor functions for now) */

        a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
        a.d1 = tmp192.d4;
        a.d2 = tmp192.d5;
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
           into tmp192.d2 which could have generated 4 carries into tmp192.d3.
           Also, since u was generated with the floor function, it could be low by up to
           almost 1.  If we account for this a value up to a.d2 could have been added into
           tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
           function, and could thus be low by almost 1.  If we account for this a value up
           to u.d2 could have been added into tmp192.d2 possibly generating a carry.
           A grand total of up to 6 carries lost. */

        mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

        tmp96.d0 = __sub_cc(b.d0, tmp96.d0);  // Compute the remainder
        tmp96.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                                 is 0 after subtraction! */
        tmp96.d2 = __subc(b.d2, tmp96.d2);
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* Since the quotient was up to 6 too small, the remainder has a maximum value of 7*f,
           or 92 bits + log2 (7) bits, which is 94.807 bits.  In theory, this kernel can handle
           f values up to 2^92.193. */

        if (shifter & 0x80000000) shl_96(&tmp96); // Optional multiply by 2.  At this point tmp96 can be 95.807 bits.

        // shifter <<= 1;
        shifter += shifter;
    }

    a.d0 = tmp96.d0;
    a.d1 = tmp96.d1;
    a.d2 = tmp96.d2;

    /* finally check if we found a factor and write the factor to RES[]
       this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
       mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
       produce an a that is too large by up to 11 times f. */
    mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}

__device__ static void test_FC96_barrett88(int96 f, int192 b, unsigned int shifter, unsigned int *RES, int bit_max64
#ifdef CPU_SIEVE
                                           ,
                                           int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           ,
                                           unsigned int *modbasecase_debug
#endif
)
{
    int96 a, u;
    int192 tmp192;
    int96 tmp96;
    float ff;

    trace_96_textmsg(__FILE__, __LINE__, f, "--- barrett88 start ---");
    /* ff = f as float, needed in mod_192_96().
       Precalculated here since it is the same for all steps in the following loop */
    ff = __uint2float_rn(f.d2);
    ff = ff * 4294967296.0f + __uint2float_rn(f.d1); /* f.d0 ignored because lower limit for this kernel are 64 bit
                                                        which yields at least 32 significant digits without f.d0! */
    ff = __int_as_float(0x3f7ffffb) / ff; // just a little bit below 1.0f so we always underestimate the quotient

    tmp192.d5 = 1 << (bit_max64 - 1); // tmp192 = 2^(95 + bits_in_f)
    tmp192.d4 = 0;
    tmp192.d3 = 0;
    tmp192.d2 = 0;
    tmp192.d1 = 0;
    tmp192.d0 = 0;

#ifndef DEBUG_GPU_MATH
    div_192_96(&u, tmp192, f, ff); // u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#else
    div_192_96(&u, tmp192, f, ff, modbasecase_debug); // u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#endif
    trace_96_96(__FILE__, __LINE__, f, "u", u);

    a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1); // a = floor(b / 2 ^ (bits_in_f - 1))
    a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);
    a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96_192_no_low3(&tmp192, a, u); /* tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)
                                          (ignore the floor functions for now) */

    a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

    a.d0 = __sub_cc(b.d0, tmp96.d0); // Compute the remainder
    a.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                         is 0 after subtraction! */
    a.d2 = __subc(b.d2, tmp96.d2);
    trace_96_96(__FILE__, __LINE__, f, "a", a);

#ifdef CPU_SIEVE
    shifter <<= 32 - shiftcount;
#endif
    while (shifter) {
        trace_96_textmsg(__FILE__, __LINE__, f, "--- main loop start ---");
        // On input a is at most 90.807 bits (see end of this loop)

        square_96_192(&b, a); // b = a^2, b is at most 181.614 bits
        trace_96_192(__FILE__, __LINE__, f, "b", b);

        if (shifter & 0x80000000) shl_192(&b); // Optional multiply by 2.  At this point b can be 182.614 bits.

        a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1); // a = b / (2 ^ (bits_in_f - 1)), a can be 95.614 bits
        a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);
        a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);
        trace_96_96(__FILE__, __LINE__, f, "a", a);

        mul_96_192_no_low3(&tmp192, a, u); /* tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)
                                              (ignore the floor functions for now) */

        a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
        a.d1 = tmp192.d4;
        a.d2 = tmp192.d5;
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
           into tmp192.d2 which could have generated 4 carries into tmp192.d3.
           Also, since u was generated with the floor function, it could be low by up to
           almost 1.  If we account for this a value up to a.d2 could have been added into
           tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
           function, and could thus be low by almost 1.  If we account for this a value up
           to u.d2 could have been added into tmp192.d2 possibly generating a carry.
           A grand total of up to 6 carries lost. */

        mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

        a.d0 = __sub_cc(b.d0, tmp96.d0);  // Compute the remainder
        a.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                             is 0 after subtraction! */
        a.d2 = __subc(b.d2, tmp96.d2);
        /* Since the quotient was up to 6 too small, the remainder has a maximum value of 7*f,
           or 88 bits + log2 (7) bits, which is 90.807 bits.  In theory, this kernel can handle
           f values up to 2^88.193. */

        // shifter <<= 1;
        shifter += shifter;
    }

/*
#ifndef DEBUG_GPU_MATH
    mod_simple_96(&a, tmp96, f, ff); // Adjustment.  The code above may produce an a that is too large by up to 6 times f.
#else
    mod_simple_96(&a, tmp96, f, ff, bit_max64 - 1, bit_max64, 6,
                  modbasecase_debug); // bit_max - 1 = bit_min (this kernel handles only single bit levels)
#endif
*/

        /* finally check if we found a factor and write the factor to RES[]
           this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
           mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
           produce an a that is too large by up to 6 times f. */
        mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}

template <int FIXED_BIT_MAX64>
__device__ static __forceinline__ void test_FC96_barrett87_impl(int96 f, int192 b, unsigned int shifter, unsigned int *RES,
                                                                int bit_max64_runtime
#ifdef CPU_SIEVE
                                                                ,
                                                                int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                                                ,
                                                                unsigned int *modbasecase_debug
#endif
)
{
    int96 a, u;
    int192 tmp192;
    int96 tmp96;
    float ff;
    const int bit_max64 = FIXED_BIT_MAX64 ? FIXED_BIT_MAX64 : bit_max64_runtime;

    trace_96_textmsg(__FILE__, __LINE__, f, "--- barrett87 start ---");
    /* ff = f as float, needed in mod_192_96().
       Precalculated here since it is the same for all steps in the following loop */
    ff = __uint2float_rn(f.d2);
    ff = ff * 4294967296.0f + __uint2float_rn(f.d1); /* f.d0 ignored because lower limit for this kernel are 64 bit
                                                        which yields at least 32 significant digits without f.d0! */
    ff = __int_as_float(0x3f7ffffb) / ff; // just a little bit below 1.0f so we always underestimate the quotient

    tmp192.d5 = 1 << (bit_max64 - 1); // tmp192 = 2^(95 + bits_in_f)
    tmp192.d4 = 0;
    tmp192.d3 = 0;
    tmp192.d2 = 0;
    tmp192.d1 = 0;
    tmp192.d0 = 0;

#ifndef DEBUG_GPU_MATH
    div_192_96(&u, tmp192, f, ff); // u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#else
    div_192_96(&u, tmp192, f, ff, modbasecase_debug); // u = floor(2^(95 + bits_in_f) / f), giving 96 bits of precision
#endif
    trace_96_96(__FILE__, __LINE__, f, "u", u);

    a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1); // a = floor(b / 2 ^ (bits_in_f - 1))
    a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);
    a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96_192_no_low3(&tmp192, a,
                       u); // tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

    a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

    // clang-format off
    a.d0 = __sub_cc( b.d0, tmp96.d0); // Compute the remainder
    a.d1 = __subc_cc(b.d1, tmp96.d1); // we do not need the upper digits of b and tmp96 because the result is 0 after subtraction!
    a.d2 = __subc(   b.d2, tmp96.d2);
    // clang-format on
    trace_96_96(__FILE__, __LINE__, f, "a", a);

#ifdef CPU_SIEVE
    shifter <<= 32 - shiftcount;
#endif
    while (shifter) {
        trace_96_textmsg(__FILE__, __LINE__, f, "--- main loop start ---");
        // On input a is at most 90.807 bits (see end of this loop)

        square_96_192(&b, a); // b = a^2, b is at most 181.614 bits
        trace_96_192(__FILE__, __LINE__, f, "b", b);

        a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1); // a = b / (2 ^ (bits_in_f - 1)), a is at most 95.614 bits
        a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);
        a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);
        trace_96_96(__FILE__, __LINE__, f, "a", a);

        mul_96_192_no_low3(&tmp192, a,
                           u); // tmp192 = (b / 2 ^ (bits_in_f - 1)) * (2 ^ (95 + bits_in_f) / f)     (ignore the floor functions for now)

        a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
        a.d1 = tmp192.d4;
        a.d2 = tmp192.d5;
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        // The quotient is off by at most 6.  A full mul_96_192 would add 5 partial results
        // into tmp192.d2 which could have generated 4 carries into tmp192.d3.
        // Also, since u was generated with the floor function, it could be low by up to
        // almost 1.  If we account for this a value up to a.d2 could have been added into
        // tmp192.d2 possibly generating a carry.  Similarly, a was generated by a floor
        // function, and could thus be low by almost 1.  If we account for this a value up
        // to u.d2 could have been added into tmp192.d2 possibly generating a carry.
        // A grand total of up to 6 carries lost.

        mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

        // clang-format off
        a.d0 = __sub_cc( b.d0, tmp96.d0); // Compute the remainder
        a.d1 = __subc_cc(b.d1, tmp96.d1); // we do not need the upper digits of b and tmp96 because the result is 0 after subtraction!
        a.d2 = __subc(   b.d2, tmp96.d2);
        // clang-format on
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        // Since the quotient was up to 6 too small, the remainder has a maximum value of 7*f,
        // or 87 bits + log2 (7) bits, which is 89.807 bits.  In theory, this kernel can handle
        // f values up to 2^87.193.

        if (shifter & 0x80000000) shl_96(&a); // "optional multiply by 2" as in Prime95 documentation
        // At this point a can be 90.807 bits.

        // shifter <<= 1;
        shifter += shifter;
    }

    /*#ifndef DEBUG_GPU_MATH
  mod_simple_96(&a, tmp96, f, ff);			// Adjustment.  The code above may produce an a that is too large by up to 12 times f.
#else
  mod_simple_96(&a, tmp96, f, ff, bit_max64 - 1, bit_max64, 11, modbasecase_debug); // bit_max - 1 = bit_min (this kernel handles only single bit levels)
#endif*/

    /* finally check if we found a factor and write the factor to RES[]
this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
produce an a that is too large by up to 11 times f. */
    mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}

__device__ static void test_FC96_barrett87(int96 f, int192 b, unsigned int shifter, unsigned int *RES, int bit_max64
#ifdef CPU_SIEVE
                                           ,
                                           int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           ,
                                           unsigned int *modbasecase_debug
#endif
)
{
#ifdef MFAKTC_BARRETT87_BIT15_SPECIAL
    if (bit_max64 == 15) {
        test_FC96_barrett87_impl<15>(f, b, shifter, RES, bit_max64
#ifdef CPU_SIEVE
                                     ,
                                     shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                     ,
                                     modbasecase_debug
#endif
        );
        return;
    }
#endif
    test_FC96_barrett87_impl<0>(f, b, shifter, RES, bit_max64
#ifdef CPU_SIEVE
                                ,
                                shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                ,
                                modbasecase_debug
#endif
    );
}

#ifdef MFAKTC_BARRETT87_GS_FIXED_SHIFTER_KERNEL
#define MFAKTC_BARRETT87_FIXED_QUOTIENT(A, TMP192, U)                                                                \
    do {                                                                                                             \
        mul_96_192_no_low3(&(TMP192), (A), (U));                                                                     \
        (A).d0 = (TMP192).d3;                                                                                        \
        (A).d1 = (TMP192).d4;                                                                                        \
        (A).d2 = (TMP192).d5;                                                                                        \
    } while (0)
#define MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT(TMP96, A, F) mul_96((TMP96), (A), (F))
#ifdef MFAKTC_BARRETT87_GS_FIXED_RECIP_NORMAL
__device__ static __forceinline__ float recip_biased_normal(float n)
{
    float r, e, t;

    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(n));
    e = __fmaf_rn(-n, r, 1.0f);
    r = __fmaf_rn(r, e, r);
    t = __fmaf_rn(r, __int_as_float(0x3f7ffffb), 0.0f);
    e = __fmaf_rn(-n, t, __int_as_float(0x3f7ffffb));
    return __fmaf_rn(r, e, t);
}
#define MFAKTC_BARRETT87_FIXED_RECIP_BIASED(N) recip_biased_normal(N)
#else
#define MFAKTC_BARRETT87_FIXED_RECIP_BIASED(N) (__int_as_float(0x3f7ffffb) / (N))
#endif

#if defined(MFAKTC_BARRETT87_GS_FIXED_SMALL_FINAL) && !defined(WAGSTAFF)
__device__ static __forceinline__ void mod_smallq_and_check_big_factor96(int96 q, int96 n, unsigned int *RES)
{
    unsigned int index, inv, low, m, n0, p0, p1, p2;

    /*
     * The fixed bit-78 Barrett path leaves q below 15*n: each loop reduction
     * is below 7*n, and the optional shifter multiply-by-two keeps it below
     * 14*n.  A factor can therefore only have q == m*n + 1 for m in [0, 14].
     */
    n0 = n.d0 & 0xfU;
    inv = n0;
    inv = (inv * (2U - n0 * inv)) & 0xfU;
    m = ((q.d0 - 1U) * inv) & 0xfU;

    if (m < 15U) {
        low = __umad32(n.d0, m, 1U);
        if (low == q.d0) {
            p1 = __add_cc(__umul32hi(n.d0, m), __umul32(n.d1, m));
            p2 = __addc(__umul32hi(n.d1, m), __umul32(n.d2, m));

            p0 = __add_cc(__umul32(n.d0, m), 1U);
            p1 = __addc_cc(p1, 0U);
            p2 = __addc(p2, 0U);

            if (q.d0 == p0 && q.d1 == p1 && q.d2 == p2) {
                index = atomicInc(&RES[0], 10000);
                if (index < 10) {
                    RES[index * 3 + 1] = n.d2;
                    RES[index * 3 + 2] = n.d1;
                    RES[index * 3 + 3] = n.d0;
                }
            }
        }
    }
}
#define MFAKTC_BARRETT87_FIXED_FINAL_CHECK(A, F, FF, RES) mod_smallq_and_check_big_factor96((A), (F), (RES))
#else
#define MFAKTC_BARRETT87_FIXED_FINAL_CHECK(A, F, FF, RES) mod_simple_96_and_check_big_factor96((A), (F), (FF), (RES))
#endif

#ifdef MFAKTC_BARRETT87_GS_FIXED_SQUARE83
__device__ static __forceinline__ void square_96_192_low83(int192 *res, int96 a)
{
    asm volatile("{\n\t"
                 ".reg .u32 a2;\n\t"

                 "mul.lo.u32      %0, %6, %6;\n\t"
                 "mul.lo.u32      %1, %6, %7;\n\t"
                 "mul.hi.u32      %2, %6, %7;\n\t"

                 "add.u32         a2, %8, %8;\n\t"

                 "add.cc.u32      %1, %1, %1;\n\t"
                 "addc.cc.u32     %2, %2, %2;\n\t"
                 "madc.hi.u32     %3, %6, a2, 0;\n\t"

                 "mad.hi.cc.u32   %1, %6, %6, %1;\n\t"
                 "madc.lo.cc.u32  %2, %7, %7, %2;\n\t"
                 "madc.hi.cc.u32  %3, %7, %7, %3;\n\t"
                 "madc.lo.u32     %4, %8, %8, 0;\n\t"

                 "mad.lo.cc.u32   %2, %6, a2, %2;\n\t"
                 "madc.lo.cc.u32  %3, %7, a2, %3;\n\t"
                 "madc.hi.cc.u32  %4, %7, a2, %4;\n\t"
                 "madc.hi.u32     %5, %8, %8, 0;\n\t"
                 "}"
                 : "=r"(res->d0), "=r"(res->d1), "=r"(res->d2), "=r"(res->d3), "=r"(res->d4), "=r"(res->d5)
                 : "r"(a.d0), "r"(a.d1), "r"(a.d2));
}
#define MFAKTC_BARRETT87_FIXED_SQUARE_96_192(RES, A) square_96_192_low83((RES), (A))
#elif defined(MFAKTC_BARRETT87_GS_FIXED_SQUARE160_GUARD)
__device__ static __forceinline__ void square_96_192_or_160_guarded(int192 *res, int96 a)
{
    if (a.d2 < 0x00010000U) {
        square_96_160(res, a);
        res->d5 = 0;
    } else {
        square_96_192(res, a);
    }
}
#define MFAKTC_BARRETT87_FIXED_SQUARE_96_192(RES, A) square_96_192_or_160_guarded((RES), (A))
#else
#define MFAKTC_BARRETT87_FIXED_SQUARE_96_192(RES, A) square_96_192((RES), (A))
#endif

template <int FIXED_BIT_MAX64, unsigned int FIXED_SHIFTER>
__device__ static __forceinline__ void test_FC96_barrett87_fixed_shifter(int96 f, int192 b, unsigned int *RES
#ifdef DEBUG_GPU_MATH
                                                                         ,
                                                                         unsigned int *modbasecase_debug
#endif
)
{
    int96 a, u;
    int192 tmp192;
    int96 tmp96;
    float ff;
    const int bit_max64 = FIXED_BIT_MAX64;

#ifdef MFAKTC_BARRETT87_GS_FIXED_INITIAL_REDUCTION
#if MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT < 0 || MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT > 95
#error MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT must be in [0, 95]
#endif
#define MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT (96 - MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT)
#endif

    trace_96_textmsg(__FILE__, __LINE__, f, "--- barrett87 fixed-shifter start ---");
    ff = __uint2float_rn(f.d2);
    ff = ff * 4294967296.0f + __uint2float_rn(f.d1);
    ff = MFAKTC_BARRETT87_FIXED_RECIP_BIASED(ff);

    tmp192.d5 = 1 << (bit_max64 - 1);
    tmp192.d4 = 0;
    tmp192.d3 = 0;
    tmp192.d2 = 0;
    tmp192.d1 = 0;
    tmp192.d0 = 0;

#ifndef DEBUG_GPU_MATH
    div_192_96(&u, tmp192, f, ff);
#else
    div_192_96(&u, tmp192, f, ff, modbasecase_debug);
#endif
    trace_96_96(__FILE__, __LINE__, f, "u", u);

#ifdef MFAKTC_BARRETT87_GS_FIXED_INITIAL_REDUCTION
#if MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT == 0
    a.d0 = u.d0;
    a.d1 = u.d1;
    a.d2 = u.d2;
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT < 32
    a.d0 = __fshift_r(u.d0, u.d1, MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT);
    a.d1 = __fshift_r(u.d1, u.d2, MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT);
    a.d2 = u.d2 >> MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT;
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT == 32
    a.d0 = u.d1;
    a.d1 = u.d2;
    a.d2 = 0;
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT < 64
    a.d0 = __fshift_r(u.d1, u.d2, MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT - 32);
    a.d1 = u.d2 >> (MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT - 32);
    a.d2 = 0;
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT == 64
    a.d0 = u.d2;
    a.d1 = 0;
    a.d2 = 0;
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT < 96
    a.d0 = u.d2 >> (MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT - 64);
    a.d1 = 0;
    a.d2 = 0;
#else
    a.d0 = 0;
    a.d1 = 0;
    a.d2 = 0;
#endif
    trace_96_96(__FILE__, __LINE__, f, "a", a);
#else
    a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1);
    a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);
    a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    MFAKTC_BARRETT87_FIXED_QUOTIENT(a, tmp192, u);
    trace_96_96(__FILE__, __LINE__, f, "a", a);
#endif

    MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT(&tmp96, a, f);
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

    // clang-format off
    a.d0 = __sub_cc( b.d0, tmp96.d0);
    a.d1 = __subc_cc(b.d1, tmp96.d1);
    a.d2 = __subc(   b.d2, tmp96.d2);
    // clang-format on
    trace_96_96(__FILE__, __LINE__, f, "a", a);

#ifdef MFAKTC_BARRETT87_GS_FIXED_INITIAL_REDUCTION
#undef MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT
#endif

#define BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(STEP, SQUARE_OP)                                                     \
    if ((FIXED_SHIFTER << (STEP)) != 0U) {                                                                            \
        trace_96_textmsg(__FILE__, __LINE__, f, "--- fixed-shifter loop step ---");                                  \
        SQUARE_OP(&b, a);                                                                                             \
        trace_96_192(__FILE__, __LINE__, f, "b", b);                                                                  \
                                                                                                                       \
        a.d0 = __fshift_r(b.d2, b.d3, bit_max64 - 1);                                                                 \
        a.d1 = __fshift_r(b.d3, b.d4, bit_max64 - 1);                                                                 \
        a.d2 = __fshift_r(b.d4, b.d5, bit_max64 - 1);                                                                 \
        trace_96_96(__FILE__, __LINE__, f, "a", a);                                                                   \
                                                                                                                       \
        MFAKTC_BARRETT87_FIXED_QUOTIENT(a, tmp192, u);                                                                \
        trace_96_96(__FILE__, __LINE__, f, "a", a);                                                                   \
                                                                                                                       \
        MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT(&tmp96, a, f);                                                            \
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);                                                          \
                                                                                                                       \
        a.d0 = __sub_cc(b.d0, tmp96.d0);                                                                              \
        a.d1 = __subc_cc(b.d1, tmp96.d1);                                                                             \
        a.d2 = __subc(b.d2, tmp96.d2);                                                                                \
        trace_96_96(__FILE__, __LINE__, f, "a", a);                                                                   \
                                                                                                                       \
        if ((FIXED_SHIFTER << (STEP)) & 0x80000000U) shl_96(&a);                                                      \
    }

    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(0, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(1, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(2, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(3, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(4, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(5, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(6, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(7, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(8, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(9, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(10, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(11, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(12, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(13, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(14, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(15, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(16, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(17, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(18, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(19, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(20, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(21, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(22, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(23, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(24, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(25, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(26, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(27, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(28, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(29, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(30, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE(31, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)

#undef BARRETT87_FIXED_SHIFTER_STEP_WITH_SQUARE

    MFAKTC_BARRETT87_FIXED_FINAL_CHECK(a, f, ff, RES);
}

#ifdef MFAKTC_BARRETT87_GS_FIXED_DUAL_CANDIDATE
#ifndef MFAKTC_BARRETT87_GS_FIXED_INITIAL_REDUCTION
#error MFAKTC_BARRETT87_GS_FIXED_DUAL_CANDIDATE requires MFAKTC_BARRETT87_GS_FIXED_INITIAL_REDUCTION
#endif

template <int FIXED_BIT_MAX64, unsigned int FIXED_SHIFTER>
__device__ static __forceinline__ void test_FC96_barrett87_fixed_shifter_dual(int96 f0, int96 f1, int192 b, unsigned int *RES
#ifdef DEBUG_GPU_MATH
                                                                              ,
                                                                              unsigned int *modbasecase_debug
#endif
)
{
    int96 a0, a1, u0, u1;
    int192 b0, b1, tmp1920, tmp1921;
    int96 tmp960, tmp961;
    float ff0, ff1;
    const int bit_max64 = FIXED_BIT_MAX64;

#if MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT < 0 || MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT > 95
#error MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT must be in [0, 95]
#endif
#define MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT (96 - MFAKTC_BARRETT87_GS_FIXED_INITIAL_A_SHIFT)

#if MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT == 0
#define MFAKTC_BARRETT87_FIXED_INITIAL_A(A, U)                                                                        \
    do {                                                                                                             \
        (A).d0 = (U).d0;                                                                                             \
        (A).d1 = (U).d1;                                                                                             \
        (A).d2 = (U).d2;                                                                                             \
    } while (0)
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT < 32
#define MFAKTC_BARRETT87_FIXED_INITIAL_A(A, U)                                                                        \
    do {                                                                                                             \
        (A).d0 = __fshift_r((U).d0, (U).d1, MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT);                            \
        (A).d1 = __fshift_r((U).d1, (U).d2, MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT);                            \
        (A).d2 = (U).d2 >> MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT;                                               \
    } while (0)
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT == 32
#define MFAKTC_BARRETT87_FIXED_INITIAL_A(A, U)                                                                        \
    do {                                                                                                             \
        (A).d0 = (U).d1;                                                                                             \
        (A).d1 = (U).d2;                                                                                             \
        (A).d2 = 0;                                                                                                  \
    } while (0)
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT < 64
#define MFAKTC_BARRETT87_FIXED_INITIAL_A(A, U)                                                                        \
    do {                                                                                                             \
        (A).d0 = __fshift_r((U).d1, (U).d2, MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT - 32);                        \
        (A).d1 = (U).d2 >> (MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT - 32);                                        \
        (A).d2 = 0;                                                                                                  \
    } while (0)
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT == 64
#define MFAKTC_BARRETT87_FIXED_INITIAL_A(A, U)                                                                        \
    do {                                                                                                             \
        (A).d0 = (U).d2;                                                                                             \
        (A).d1 = 0;                                                                                                  \
        (A).d2 = 0;                                                                                                  \
    } while (0)
#elif MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT < 96
#define MFAKTC_BARRETT87_FIXED_INITIAL_A(A, U)                                                                        \
    do {                                                                                                             \
        (A).d0 = (U).d2 >> (MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT - 64);                                        \
        (A).d1 = 0;                                                                                                  \
        (A).d2 = 0;                                                                                                  \
    } while (0)
#else
#define MFAKTC_BARRETT87_FIXED_INITIAL_A(A, U)                                                                        \
    do {                                                                                                             \
        (A).d0 = 0;                                                                                                  \
        (A).d1 = 0;                                                                                                  \
        (A).d2 = 0;                                                                                                  \
    } while (0)
#endif

    ff0 = __uint2float_rn(f0.d2);
    ff0 = ff0 * 4294967296.0f + __uint2float_rn(f0.d1);
    ff0 = MFAKTC_BARRETT87_FIXED_RECIP_BIASED(ff0);

    ff1 = __uint2float_rn(f1.d2);
    ff1 = ff1 * 4294967296.0f + __uint2float_rn(f1.d1);
    ff1 = MFAKTC_BARRETT87_FIXED_RECIP_BIASED(ff1);

#define MFAKTC_BARRETT87_FIXED_INIT_TMP192(TMP192)                                                                    \
    do {                                                                                                             \
        (TMP192).d5 = 1 << (bit_max64 - 1);                                                                          \
        (TMP192).d4 = 0;                                                                                             \
        (TMP192).d3 = 0;                                                                                             \
        (TMP192).d2 = 0;                                                                                             \
        (TMP192).d1 = 0;                                                                                             \
        (TMP192).d0 = 0;                                                                                             \
    } while (0)

    MFAKTC_BARRETT87_FIXED_INIT_TMP192(tmp1920);
#ifndef DEBUG_GPU_MATH
    div_192_96(&u0, tmp1920, f0, ff0);
#else
    div_192_96(&u0, tmp1920, f0, ff0, modbasecase_debug);
#endif

    MFAKTC_BARRETT87_FIXED_INIT_TMP192(tmp1921);
#ifndef DEBUG_GPU_MATH
    div_192_96(&u1, tmp1921, f1, ff1);
#else
    div_192_96(&u1, tmp1921, f1, ff1, modbasecase_debug);
#endif

#undef MFAKTC_BARRETT87_FIXED_INIT_TMP192

    b0 = b;
    b1 = b;
    MFAKTC_BARRETT87_FIXED_INITIAL_A(a0, u0);
    MFAKTC_BARRETT87_FIXED_INITIAL_A(a1, u1);

    MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT(&tmp960, a0, f0);
    MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT(&tmp961, a1, f1);

    a0.d0 = __sub_cc(b0.d0, tmp960.d0);
    a0.d1 = __subc_cc(b0.d1, tmp960.d1);
    a0.d2 = __subc(b0.d2, tmp960.d2);

    a1.d0 = __sub_cc(b1.d0, tmp961.d0);
    a1.d1 = __subc_cc(b1.d1, tmp961.d1);
    a1.d2 = __subc(b1.d2, tmp961.d2);

#undef MFAKTC_BARRETT87_FIXED_INITIAL_A
#undef MFAKTC_BARRETT87_GS_FIXED_INITIAL_U_RSHIFT

#define BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(STEP, SQUARE_OP)                                                       \
    if ((FIXED_SHIFTER << (STEP)) != 0U) {                                                                            \
        SQUARE_OP(&b0, a0);                                                                                           \
        SQUARE_OP(&b1, a1);                                                                                           \
                                                                                                                       \
        a0.d0 = __fshift_r(b0.d2, b0.d3, bit_max64 - 1);                                                             \
        a0.d1 = __fshift_r(b0.d3, b0.d4, bit_max64 - 1);                                                             \
        a0.d2 = __fshift_r(b0.d4, b0.d5, bit_max64 - 1);                                                             \
        a1.d0 = __fshift_r(b1.d2, b1.d3, bit_max64 - 1);                                                             \
        a1.d1 = __fshift_r(b1.d3, b1.d4, bit_max64 - 1);                                                             \
        a1.d2 = __fshift_r(b1.d4, b1.d5, bit_max64 - 1);                                                             \
                                                                                                                       \
        MFAKTC_BARRETT87_FIXED_QUOTIENT(a0, tmp1920, u0);                                                            \
        MFAKTC_BARRETT87_FIXED_QUOTIENT(a1, tmp1921, u1);                                                            \
                                                                                                                       \
        MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT(&tmp960, a0, f0);                                                        \
        MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT(&tmp961, a1, f1);                                                        \
                                                                                                                       \
        a0.d0 = __sub_cc(b0.d0, tmp960.d0);                                                                          \
        a0.d1 = __subc_cc(b0.d1, tmp960.d1);                                                                         \
        a0.d2 = __subc(b0.d2, tmp960.d2);                                                                            \
        a1.d0 = __sub_cc(b1.d0, tmp961.d0);                                                                          \
        a1.d1 = __subc_cc(b1.d1, tmp961.d1);                                                                         \
        a1.d2 = __subc(b1.d2, tmp961.d2);                                                                            \
                                                                                                                       \
        if ((FIXED_SHIFTER << (STEP)) & 0x80000000U) {                                                               \
            shl_96(&a0);                                                                                             \
            shl_96(&a1);                                                                                             \
        }                                                                                                            \
    }

    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(0, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(1, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(2, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(3, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(4, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(5, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(6, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(7, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(8, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(9, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(10, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(11, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(12, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(13, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(14, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(15, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(16, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(17, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(18, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(19, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(20, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(21, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(22, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(23, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(24, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(25, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(26, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(27, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(28, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(29, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(30, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)
    BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE(31, MFAKTC_BARRETT87_FIXED_SQUARE_96_192)

#undef BARRETT87_FIXED_DUAL_STEP_WITH_SQUARE

    MFAKTC_BARRETT87_FIXED_FINAL_CHECK(a0, f0, ff0, RES);
    MFAKTC_BARRETT87_FIXED_FINAL_CHECK(a1, f1, ff1, RES);
}
#endif

#undef MFAKTC_BARRETT87_FIXED_MUL_QUOTIENT
#undef MFAKTC_BARRETT87_FIXED_QUOTIENT
#undef MFAKTC_BARRETT87_FIXED_SQUARE_96_192
#undef MFAKTC_BARRETT87_FIXED_RECIP_BIASED
#undef MFAKTC_BARRETT87_FIXED_FINAL_CHECK
#endif

__device__ static void test_FC96_barrett79(int96 f, int192 b, unsigned int shifter, unsigned int *RES
#ifdef CPU_SIEVE
                                           ,
                                           int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           ,
                                           int bit_max64, unsigned int *modbasecase_debug
#endif
)
{
    int96 a, u;
    int192 tmp192;
    int96 tmp96;
    float ff;

    trace_96_textmsg(__FILE__, __LINE__, f, "--- barrett79 start ---");
    /*
ff = f as float, needed in mod_160_96().
Precalculated here since it is the same for all steps in the following loop */
    ff = __uint2float_rn(f.d2);
    ff = ff * 4294967296.0f + __uint2float_rn(f.d1); /* f.d0 ignored because lower limit for this kernel are 64 bit
                                                        which yields at least 32 significant digits without f.d0! */
    ff = __int_as_float(0x3f7ffffb) / ff; // just a little bit below 1.0f so we always underestimate the quotient

#ifndef DEBUG_GPU_MATH
    inv_160_96(&u, f, ff); // u = floor(2^160 / f)
#else
    inv_160_96(&u, f, ff, modbasecase_debug); // u = floor(2^160 / f)
#endif
    trace_96_96(__FILE__, __LINE__, f, "u", u);

    a.d0 = b.d2; // a = floor(b / 2^64)
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3(&tmp192, a, u); // tmp192 = (b / 2^64) * (2 ^ 160 / f) (ignore the floor functions for now)

    a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

    // clang-format off
    tmp96.d0 = __sub_cc( b.d0, tmp96.d0); // Compute the remainder
    tmp96.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                             is 0 after subtraction! */
    tmp96.d2 = __subc(   b.d2, tmp96.d2);
    // clang-format on
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

#ifdef CPU_SIEVE
    shifter <<= 32 - shiftcount;
#endif
    while (shifter) {
        trace_96_textmsg(__FILE__, __LINE__, f, "--- main loop start ---");
#ifndef DEBUG_GPU_MATH
        mod_simple_96(&a, tmp96, f, ff); /* Adjustment. The code above/below may produce an a
                                            that is too large by up to 11 times f. */
#else
        mod_simple_96(&a, tmp96, f, ff, 0, 79 - 64, 10, modbasecase_debug);
#endif
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* Since mod_simple_96 does not do a complete adjustment we need to allow one bit
           for that.  Thus, at this point a can be 80 bits.

           On input a is at most 79 bits (see mod_simple_96 above) */

        square_96_160(&b, a); // b = a^2, b is at most 158 bits

        a.d0 = b.d2; // a = floor (b / 2^64)
        a.d1 = b.d3;
        a.d2 = b.d4;
        trace_96_96(__FILE__, __LINE__, f, "a", a);

        mul_96_192_no_low3(&tmp192, a, u); // tmp192 = (b / 2^64) * (2 ^ 160 / f) (ignore the floor functions for now)

        a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
        a.d1 = tmp192.d4;
        a.d2 = tmp192.d5;
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* The quotient is off by at most 5.  A full mul_96_192 would add 5 partial results
           into tmp192.d2 which could have generated 4 carries into tmp192.d3.
           Also, since u was generated with the floor function, it could be low by up to
           almost 1.  If we account for this a value up to a.d2 could have been added into
           tmp192.d2.  Since we know the maximum value of b, the maximum value of a.d2
           is 2^30.  Similarly, a was generated by a floor function, and could thus be
           low by almost 1.  If we account for this a value up to u.d2 could have been added
           into tmp192.d2.  Since we know the maximum value of f is 79 bits, the maximum value
           of u is 160-79 (81) bits.  Thus the maximum value of u.d2 is 2^17.
           Since maximum a.d2 + maximum u.d2 is less than 2^32, these 2 values combined can
           only generate 1 carry into tmp192.d3 -- for a total of up to 5 carries lost. */

        mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

        // clang-format off
        tmp96.d0 = __sub_cc( b.d0, tmp96.d0); // Compute the remainder
        tmp96.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                                 is 0 after subtraction! */
        tmp96.d2 = __subc(   b.d2, tmp96.d2);
        // clang-format on
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);
        /* Since the quotient was up to 5 too small, the remainder has a maximum value of 6*f,
           or 79 bits + log2 (6) bits, which is 81.585 bits.  In theory, this kernel can handle
           f values up to 2^79.415. */

        if (shifter & 0x80000000) shl_96(&tmp96); // "optional multiply by 2" as in Prime95 documentation
        // At this point a can be 82.585 bits.

        // shifter <<= 1;
        shifter += shifter;
    }

    a.d0 = tmp96.d0;
    a.d1 = tmp96.d1;
    a.d2 = tmp96.d2;

    /* finally check if we found a factor and write the factor to RES[]
       this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
       mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
       produce an a that is too large by up to 11 times f. */
    mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}

__device__ static void test_FC96_barrett77(int96 f, int192 b, unsigned int shifter, unsigned int *RES
#ifdef CPU_SIEVE
                                           ,
                                           int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           ,
                                           int bit_max64, unsigned int *modbasecase_debug
#endif
)
{
    int96 a, u;
    int192 tmp192;
    int96 tmp96;
    float ff;

    trace_96_textmsg(__FILE__, __LINE__, f, "--- barrett77 start ---");
    /* ff = f as float, needed in mod_160_96().
       Precalculated here since it is the same for all steps in the following loop */
    ff = __uint2float_rn(f.d2);
    ff = ff * 4294967296.0f + __uint2float_rn(f.d1); /* f.d0 ignored because lower limit for this kernel are 64 bit
                                                        which yields at least 32 significant digits without f.d0! */
    ff = __int_as_float(0x3f7ffffb) / ff; // just a little bit below 1.0f so we always underestimate the quotient

#ifndef DEBUG_GPU_MATH
    inv_160_96(&u, f, ff); // u = floor(2^160 / f)
#else
    inv_160_96(&u, f, ff, modbasecase_debug); // u = floor(2^160 / f)
#endif
    trace_96_96(__FILE__, __LINE__, f, "u", u);

    a.d0 = b.d2; // a = floor(b / 2^64)
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3(&tmp192, a, u); // tmp192 = (b / 2^64) * (2 ^ 160 / f) (ignore the floor functions for now)

    a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

    // clang-format off
    a.d0 = __sub_cc(b.d0,  tmp96.d0); // Compute the remainder
    a.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                         is 0 after subtraction! */
    a.d2 = __subc(b.d2,    tmp96.d2);
    // clang-format on
    trace_96_96(__FILE__, __LINE__, f, "a", a);

#ifdef DEBUG_GPU_MATH
    if (f.d2) // check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
    {
        MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2,
                                    13) // a should never have a value >= 2^80, if so square_96_160() will overflow!
    } // this will warn whenever a becomes close to 2^80
#endif

#ifdef CPU_SIEVE
    shifter <<= 32 - shiftcount;
#endif
    while (shifter) {
        trace_96_textmsg(__FILE__, __LINE__, f, "--- main loop start ---");
        // On input a is at most 79.322 bits (see end of this loop)

        square_96_160(&b, a); // b = a^2, b is at most 158.644 bits

        if (shifter & 0x80000000) shl_192(&b); // Optional multiply by 2. At this point b can be 159.644 bits.

        a.d0 = b.d2; // a = floor (b / 2^64)
        a.d1 = b.d3;
        a.d2 = b.d4;
        trace_96_96(__FILE__, __LINE__, f, "a", a);

        mul_96_192_no_low3_special(&tmp192, a, u); /* tmp192 = (b / 2^64) * (2 ^ 160 / f)
                                                      (ignore the floor functions for now) */

        a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
        a.d1 = tmp192.d4;
        a.d2 = tmp192.d5;
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* In the case we care about most (large f values that might cause b to exceed 160 bits),
           the quotient is off by at most 4.  A full mul_96_192 would add 5 partial results
           into tmp192.d2, whereas mul_96_192_no_low3_special adds only 2 partial results,
           which could have generated 3 more carries into tmp192.d3.
           Also, since u was generated with the floor function, it could be low by up to
           almost 1.  If we account for this a value up to a.d2 could have been added into
           tmp192.d2.  Since we know the maximum value of b, the maximum value of a.d2
           is 2^31.17.  Similarly, a was generated by a floor function, and could thus be
           low by almost 1.  If we account for this a value up to u.d2 could have been added
           into tmp192.d2.  Since we know the maximum value of f is 77 bits, the maximum value
           of u is 160-77 (83) bits.  Thus the maximum value of u.d2 is 2^19.
           Since maximum a.d2 + maximum u.d2 is less than 2^32, these 2 values combined can
           only generate only 1 carry into tmp192.d3 -- for a total of up to 4 carries lost. */

        mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

        // clang-format off
        a.d0 = __sub_cc( b.d0, tmp96.d0); // Compute the remainder
        a.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                             is 0 after subtraction! */
        a.d2 = __subc(   b.d2, tmp96.d2);
        // clang-format on
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* Since the quotient was up to 4 too small, the remainder has a maximum value of 5*f,
           or 77 bits + log2 (5) bits, which is 79.322 bits.  In theory, this kernel can handle
           f values up to 2^77.178. */

#ifdef DEBUG_GPU_MATH
        if (f.d2) // check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
        {
            MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2, 13) /* a should never have a value >= 2^80, 
                                                                         if so square_96_160() will overflow! */
        } // this will warn whenever a becomes close to 2^80
#endif

        // shifter <<= 1;
        shifter += shifter;
    }

/*
#ifndef DEBUG_GPU_MATH
    mod_simple_96(&a, tmp96, f, ff); // Adjustment.  The code above may produce an a that is too large by up to 5 times f.
#else
    mod_simple_96(&a, tmp96, f, ff, 0, 79 - 64, 4, modbasecase_debug);
#endif
*/

        /* finally check if we found a factor and write the factor to RES[]
           this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
           mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
           produce an a that is too large by up to 5 times f. */
        mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}

__device__ static void test_FC96_barrett76(int96 f, int192 b, unsigned int shifter, unsigned int *RES
#ifdef CPU_SIEVE
                                           ,
                                           int shiftcount
#endif
#ifdef DEBUG_GPU_MATH
                                           ,
                                           int bit_max64, unsigned int *modbasecase_debug
#endif
)
{
    int96 a, u;
    int192 tmp192;
    int96 tmp96;
    float ff;

    trace_96_textmsg(__FILE__, __LINE__, f, "--- barrett76 start ---");
    /* ff = f as float, needed in mod_160_96().
       Precalculated here since it is the same for all steps in the following loop */
    ff = __uint2float_rn(f.d2);
    ff = ff * 4294967296.0f + __uint2float_rn(f.d1); /* f.d0 ignored because lower limit for this kernel are 64 bit
                                                        which yields at least 32 significant digits without f.d0! */
    ff = __int_as_float(0x3f7ffffb) / ff; // just a little bit below 1.0f so we always underestimate the quotient

#ifndef DEBUG_GPU_MATH
    inv_160_96(&u, f, ff); // u = floor(2^160 / f)
#else
    inv_160_96(&u, f, ff, modbasecase_debug); // u = floor(2^160 / f)
#endif
    trace_96_96(__FILE__, __LINE__, f, "u", u);

    a.d0 = b.d2; // a = floor(b / 2^64)
    a.d1 = b.d3;
    a.d2 = b.d4;

    mul_96_192_no_low3(&tmp192, a, u); // tmp192 = (b / 2^64) * (2 ^ 160 / f) (ignore the floor functions for now)

    a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
    a.d1 = tmp192.d4;
    a.d2 = tmp192.d5;
    trace_96_96(__FILE__, __LINE__, f, "a", a);

    mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
    trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

    // clang-format off
    a.d0 = __sub_cc( b.d0, tmp96.d0); // Compute the remainder
    a.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                         is 0 after subtraction! */
    a.d2 = __subc(   b.d2, tmp96.d2);
    // clang-format on
    trace_96_96(__FILE__, __LINE__, f, "a", a);

#ifdef DEBUG_GPU_MATH
    if (f.d2) // check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
    {
        MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2, 13) /* a should never have a value >= 2^80,
                                                                     if so square_96_160() will overflow! */
    } // this will warn whenever a becomes close to 2^80
#endif

#ifdef CPU_SIEVE
    shifter <<= 32 - shiftcount;
#endif
    while (shifter) {
        trace_96_textmsg(__FILE__, __LINE__, f, "--- main loop start ---");
        // On input a is at most 79.585 bits (see end of this loop)

        square_96_160(&b, a); // b = a^2, b is at most 159.17 bits

        a.d0 = b.d2; // a = floor (b / 2^64)
        a.d1 = b.d3;
        a.d2 = b.d4;
        trace_96_96(__FILE__, __LINE__, f, "a", a);

        mul_96_192_no_low3(&tmp192, a, u); // tmp192 = (b / 2^64) * (2 ^ 160 / f) (ignore the floor functions for now)

        a.d0 = tmp192.d3; // a = tmp192 / 2^96, which if we do the math simplifies to the quotient: b / f
        a.d1 = tmp192.d4;
        a.d2 = tmp192.d5;
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* In the case we care about most (large f values that might cause b to exceed 160 bits),
           the quotient is off by at most 5.  A full mul_96_192 would add 5 partial results
           into tmp192.d2 which could have generated 4 carries into tmp192.d3.
           Also, since u was generated with the floor function, it could be low by up to
           almost 1.  If we account for this a value up to a.d2 could have been added into
           tmp192.d2.  Since we know the maximum value of b, the maximum value of a.d2
           is 2^31.17.  Similarly, a was generated by a floor function, and could thus be
           low by almost 1.  If we account for this a value up to u.d2 could have been added
           into tmp192.d2.  Since we know the maximum value of f is 76 bits, the maximum value
           of u is 160-76 (84) bits.  Thus the maximum value of u.d2 is 2^20.
           Since maximum a.d2 + maximum u.d2 is less than 2^32, these 2 values combined can
           only generate only 1 carry into tmp192.d3 -- for a total of up to 5 carries lost. */

        mul_96(&tmp96, a, f); // tmp96 = quotient * f, we only compute the low 96-bits here
        trace_96_96(__FILE__, __LINE__, f, "tmp96", tmp96);

        // clang-format off
        a.d0 = __sub_cc( b.d0, tmp96.d0); // Compute the remainder
        a.d1 = __subc_cc(b.d1, tmp96.d1); /* we do not need the upper digits of b and tmp96 because the result
                                             is 0 after subtraction! */
        a.d2 = __subc(   b.d2, tmp96.d2);
        // clang-format on
        trace_96_96(__FILE__, __LINE__, f, "a", a);
        /* Since the quotient was up to 5 too small, the remainder has a maximum value of 6*f,
           or 76 bits + log2 (6) bits, which is 78.585 bits.  In theory, this kernel can handle
           f values up to 2^76.415. */

        if (shifter & 0x80000000) shl_96(&a); // "optional multiply by 2" as in Prime95 documentation
        // At this point a can be 79.585 bits.
        trace_96_96(__FILE__, __LINE__, f, "a", a);

#ifdef DEBUG_GPU_MATH
        if (f.d2) // check only when f is >= 2^64 (f <= 2^64 is not supported by this kernel
        {
            MODBASECASE_VALUE_BIG_ERROR(0xC000, "a.d2", 99, a.d2, 13) /* a should never have a value >= 2^80,
                                                                         if so square_96_160() will overflow! */
        } // this will warn whenever a becomes close to 2^80
#endif

        // shifter <<= 1;
        shifter += shifter;
    }

/*
#ifndef DEBUG_GPU_MATH
    mod_simple_96(&a, tmp96, f, ff); // Adjustment.  The code above may produce an a that is too large by up to 11 times f.
#else
    mod_simple_96(&a, tmp96, f, ff, 0, 79 - 64, 11, modbasecase_debug);
#endif
*/

        /* finally check if we found a factor and write the factor to RES[]
           this kernel has a lower FC limit of 2^64 so we can use [mod_simple_96_and_]check_big_factor96().
           mod_simple_96_and_check_big_factor96() includes the final adjustment, too. The code above may
           produce an a that is too large by up to 11 times f. */
        mod_simple_96_and_check_big_factor96(a, f, ff, RES);
}
