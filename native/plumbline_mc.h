/* Plumbline -- C ABI for the native Monte Carlo backend.
 *
 * This header is the contract between the C++ engine and the ctypes loader in
 * plumbline/engines/native.py. Two rules keep that contract cheap to hold:
 *
 *   1. Every field is eight bytes wide. There is therefore no padding on any
 *      supported platform, and the ctypes Structure cannot silently disagree
 *      with the compiler about the layout.
 *   2. The first field of each struct is its own size. The loader checks it
 *      before the first call, so a stale shared library is caught at load time
 *      rather than as unexplained numbers later.
 *
 * The library is deliberately not a Python extension module. It exports plain
 * C, so one build works for every interpreter version and the build needs no
 * Python headers.
 */

#ifndef PLUMBLINE_MC_H
#define PLUMBLINE_MC_H

#include <stdint.h>

/* The build compiles with -fvisibility=hidden, which is the right default for
 * a shared library but hides these too unless each one is marked. Windows
 * needs the opposite treatment: MinGW auto-exports, MSVC exports nothing
 * without __declspec. One macro covers all three platforms. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef PLUMBLINE_BUILDING
#    define PLUMBLINE_API __declspec(dllexport)
#  else
#    define PLUMBLINE_API
#  endif
#else
#  define PLUMBLINE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* instrument codes, matching plumbline.engines.native.INSTRUMENT_CODES */
#define PLUMBLINE_EUROPEAN 0
#define PLUMBLINE_ASIAN    1
#define PLUMBLINE_BARRIER  2
#define PLUMBLINE_DIGITAL  3
#define PLUMBLINE_LOOKBACK 4

/* status codes returned by plumbline_mc_price */
#define PLUMBLINE_OK                 0
#define PLUMBLINE_BAD_STRUCT_SIZE    1
#define PLUMBLINE_BAD_INSTRUMENT     2
#define PLUMBLINE_BAD_PARAMETER      3
#define PLUMBLINE_DEGENERATE_INPUT   4

typedef struct PlumblineMCRequest {
    int64_t struct_size;      /* sizeof(PlumblineMCRequest) */
    int64_t instrument;       /* one of the PLUMBLINE_* instrument codes */
    int64_t is_call;          /* 1 for a call, 0 for a put */

    double  spot;
    double  strike;
    double  maturity;         /* years */
    double  rate;             /* continuously compounded */
    double  dividend;         /* continuously compounded */
    double  volatility;

    /* barrier */
    double  barrier;
    int64_t barrier_is_down;  /* 1 down, 0 up */
    int64_t barrier_is_out;   /* 1 knock-out, 0 knock-in */

    /* asian */
    int64_t averaging_arithmetic; /* 1 arithmetic, 0 geometric */

    /* digital */
    int64_t payout_asset;     /* 1 asset-or-nothing, 0 cash-or-nothing */
    double  cash_amount;

    /* lookback */
    int64_t strike_fixed;     /* 1 fixed strike, 0 floating strike */

    /* estimator */
    int64_t paths;
    int64_t steps;
    uint64_t seed;
    int64_t antithetic;       /* 1 to pair every draw with its negation */
    int64_t control_variate;  /* 1 to subtract the control */
    double  control_mean;     /* E[control], supplied by the caller */

    int64_t threads;          /* 0 asks the library to choose */
    int64_t block_pairs;      /* path-pairs per RNG block; 0 takes the default */
} PlumblineMCRequest;

typedef struct PlumblineMCResult {
    int64_t struct_size;      /* sizeof(PlumblineMCResult) */
    double  price;
    double  standard_error;
    double  control_beta;
    int64_t paths;            /* paths actually simulated */
    int64_t threads;          /* threads actually used */
    int64_t blocks;           /* RNG blocks consumed */
} PlumblineMCResult;

/* Price one contract. Returns a PLUMBLINE_* status code. The result is only
 * meaningful when the return value is PLUMBLINE_OK. */
PLUMBLINE_API int32_t plumbline_mc_price(const PlumblineMCRequest *request,
                           PlumblineMCResult *result);

/* Build identity, for the Audit Report and for the loader's sanity check. */
PLUMBLINE_API const char *plumbline_backend_version(void);

/* Hardware concurrency the library sees, or 1 when it cannot tell. */
PLUMBLINE_API int64_t plumbline_backend_threads(void);

/* Struct sizes the library was compiled with, for the loader's ABI check. */
PLUMBLINE_API int64_t plumbline_request_size(void);
PLUMBLINE_API int64_t plumbline_result_size(void);

#ifdef __cplusplus
}
#endif

#endif /* PLUMBLINE_MC_H */
