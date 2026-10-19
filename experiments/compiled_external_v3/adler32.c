/* Altered, standalone benchmark extraction from zlib v1.3.1 adler32.c.
 * Copyright (C) 1995-2011, 2016 Mark Adler
 * Original blob: 04b81d29bad1691878648d7ae1680726abd905e4.
 * See LICENSE.md for the unaltered zlib distribution notice.
 * Changes: replace zutil.h with type definitions; retain division-enabled
 * macros and adler32_z; omit combine APIs; append a bounded entry wrapper.
 * The adler32_z function body below is unchanged from the cited version.
 */
typedef unsigned long uLong;
typedef unsigned char Bytef;
typedef unsigned long z_size_t;
typedef unsigned int u32;
#define ZEXPORT
#define Z_NULL 0
#define BASE 65521U
#define NMAX 5552
#define DO1(buf,i)  {adler += (buf)[i]; sum2 += adler;}
#define DO2(buf,i)  DO1(buf,i); DO1(buf,i+1);
#define DO4(buf,i)  DO2(buf,i); DO2(buf,i+2);
#define DO8(buf,i)  DO4(buf,i); DO4(buf,i+4);
#define DO16(buf)   DO8(buf,0); DO8(buf,8);
#define MOD(a) a %= BASE
#define MOD28(a) a %= BASE

uLong ZEXPORT adler32_z(uLong adler, const Bytef *buf, z_size_t len) {
    unsigned long sum2;
    unsigned n;

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1) {
        adler += buf[0];
        if (adler >= BASE)
            adler -= BASE;
        sum2 += adler;
        if (sum2 >= BASE)
            sum2 -= BASE;
        return adler | (sum2 << 16);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == Z_NULL)
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (len < 16) {
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);            /* only added so many BASE's */
        return adler | (sum2 << 16);
    }

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        len -= NMAX;
        n = NMAX / 16;          /* NMAX is divisible by 16 */
        do {
            DO16(buf);          /* 16 sums unrolled */
            buf += 16;
        } while (--n);
        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        while (len >= 16) {
            len -= 16;
            DO16(buf);
            buf += 16;
        }
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        MOD(adler);
        MOD(sum2);
    }

    /* return recombined sums */
    return adler | (sum2 << 16);
}

/* Original benchmark wrapper, not part of upstream zlib. The interface's n is
 * an input-word count, so the checksum length is 4*n, bounded to 0..128 bytes.
 * The long NMAX path remains in the compiled code but is outside this envelope.
 */
u32 entry(u32 *a,u32 n,u32 key,u32 *b,u32 *out){
    (void)b; (void)out;
    return (u32)adler32_z((uLong)key,(const Bytef*)a,(z_size_t)n*4u);
}
