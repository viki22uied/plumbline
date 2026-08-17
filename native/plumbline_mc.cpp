/* Plumbline -- native Monte Carlo backend.
 *
 * This is an optional backend. The NumPy engine in plumbline/engines/
 * montecarlo.py stays the documented default and the reference for what the
 * estimator means. This file exists to do the same arithmetic faster, and it
 * must not do it differently.
 *
 * Three things it does that the vectorised version cannot:
 *
 *   1. One pass per path. NumPy walks a length-N array once per operation per
 *      time step, so a barrier simulation touches several arrays of N doubles
 *      at every step and the working set leaves cache. Here one path-pair is
 *      carried in registers from t=0 to expiry.
 *   2. Real threads. The path loop is embarrassingly parallel and there is no
 *      interpreter lock in the way.
 *   3. No temporaries. The vectorised barrier step allocates a handful of
 *      length-N arrays per time step; this allocates nothing in the loop.
 *
 * What it deliberately does NOT do is reproduce NumPy's random stream. A
 * different generator makes this a second, independent estimator of the same
 * expectation, which is the more useful thing to have in a validation tool:
 * the two backends agreeing within their combined standard error is evidence,
 * whereas two bit-identical numbers would only prove that one copied the
 * other. The parity tests assert exactly that, and both are separately pinned
 * against the closed forms.
 *
 * Determinism does not depend on the thread count. Path-pairs are cut into
 * fixed-size blocks, block k always draws from stream k, and threads pull
 * blocks off an atomic counter. One thread and twelve threads give the same
 * answer to the last bit.
 */

#include "plumbline_mc.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#define PLUMBLINE_BACKEND_VERSION "plumbline-mc 1.0.0"

namespace {

constexpr int64_t kDefaultBlockPairs = 2048;
constexpr double kTwoPi = 6.283185307179586476925286766559;

/* ---------------------------------------------------------------------------
 * Random numbers
 *
 * xoshiro256++ seeded through splitmix64, which is what the generator's
 * authors recommend. Only 64-bit arithmetic is used, so there is no
 * __uint128_t and the same code compiles under MSVC as under GCC and Clang.
 * ------------------------------------------------------------------------ */

inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

private:
    uint64_t state_;
};

class Xoshiro256pp {
public:
    /* Block k of a run seeded with `seed` always gets the same state, so the
     * result does not depend on how the blocks were shared out. */
    Xoshiro256pp(uint64_t seed, uint64_t block) {
        SplitMix64 seeder(seed ^ (block * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL));
        for (int i = 0; i < 4; ++i) {
            state_[i] = seeder.next();
        }
        has_spare_normal_ = false;
        spare_normal_ = 0.0;
    }

    uint64_t next_bits() {
        const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    /* Uniform on (0, 1). The lower bound is open, because the bridge sampler
     * takes a logarithm of it. */
    double next_uniform() {
        const double u = static_cast<double>(next_bits() >> 11) * 0x1.0p-53;
        return u > 0.0 ? u : 0x1.0p-53;
    }

    /* Box-Muller. Two normals come out of every pair of uniforms, so the
     * second is kept rather than thrown away. */
    double next_normal() {
        if (has_spare_normal_) {
            has_spare_normal_ = false;
            return spare_normal_;
        }
        const double u1 = next_uniform();
        const double u2 = next_uniform();
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = kTwoPi * u2;
        spare_normal_ = radius * std::sin(angle);
        has_spare_normal_ = true;
        return radius * std::cos(angle);
    }

private:
    uint64_t state_[4];
    double spare_normal_;
    bool has_spare_normal_;
};

/* ---------------------------------------------------------------------------
 * Streaming moments
 *
 * Welford, with Chan's merge rule, so the partial results of independent
 * threads combine exactly and the variance never comes from the cancelling
 * difference of two large sums.
 * ------------------------------------------------------------------------ */

struct Accumulator {
    int64_t count = 0;
    double mean_y = 0.0;
    double mean_x = 0.0;
    double m2_y = 0.0;
    double m2_x = 0.0;
    double comoment = 0.0;

    void push(double y, double x) {
        count += 1;
        const double inv = 1.0 / static_cast<double>(count);
        const double dy = y - mean_y;
        const double dx = x - mean_x;
        mean_y += dy * inv;
        mean_x += dx * inv;
        m2_y += dy * (y - mean_y);
        m2_x += dx * (x - mean_x);
        comoment += dx * (y - mean_y);
    }

    void merge(const Accumulator &other) {
        if (other.count == 0) {
            return;
        }
        if (count == 0) {
            *this = other;
            return;
        }
        const double na = static_cast<double>(count);
        const double nb = static_cast<double>(other.count);
        const double total = na + nb;
        const double dy = other.mean_y - mean_y;
        const double dx = other.mean_x - mean_x;
        const double weight = na * nb / total;

        mean_y += dy * nb / total;
        mean_x += dx * nb / total;
        m2_y += other.m2_y + dy * dy * weight;
        m2_x += other.m2_x + dx * dx * weight;
        comoment += other.comoment + dx * dy * weight;
        count += other.count;
    }
};

/* ---------------------------------------------------------------------------
 * Per-path simulation
 * ------------------------------------------------------------------------ */

struct Contract {
    int64_t instrument;
    double phi;              /* +1 call, -1 put */
    double spot;
    double strike;
    double maturity;
    double discount;         /* exp(-r T) */
    double drift_total;      /* (r - q - v^2/2) T, terminal-only instruments */
    double vol_total;        /* sigma sqrt(T) */

    /* stepped instruments */
    int64_t steps;
    double drift_step;
    double vol_step;
    double var_step;

    /* barrier */
    double log_barrier;
    bool barrier_is_down;
    bool barrier_is_out;

    /* asian */
    bool averaging_arithmetic;

    /* digital */
    bool payout_asset;
    double cash_amount;

    /* lookback */
    bool strike_fixed;
};

inline double positive_part(double x) { return x > 0.0 ? x : 0.0; }

/* Terminal-only instruments: one normal decides the whole path. */
inline void terminal_sample(const Contract &c, double z, double &payoff, double &control) {
    const double terminal = c.spot * std::exp(c.drift_total + c.vol_total * z);
    control = c.discount * terminal;

    if (c.instrument == PLUMBLINE_EUROPEAN) {
        payoff = c.discount * positive_part(c.phi * (terminal - c.strike));
        return;
    }
    /* digital */
    const bool in_the_money =
        c.phi > 0.0 ? (terminal > c.strike) : (terminal < c.strike);
    if (!in_the_money) {
        payoff = 0.0;
    } else if (c.payout_asset) {
        payoff = c.discount * terminal;
    } else {
        payoff = c.discount * c.cash_amount;
    }
}

/* Stepped instruments. `sign` is +1 for the ordinary draw and -1 for its
 * antithetic partner; both halves of a pair walk the same normals. */
inline void stepped_sample(const Contract &c, Xoshiro256pp &rng_uniforms,
                           double sign, const std::vector<double> &normals,
                           double &payoff, double &control) {
    const double log_spot = std::log(c.spot);
    double x = log_spot;
    double arithmetic_sum = 0.0;
    double geometric_sum = 0.0;
    double running_max = log_spot;
    double running_min = log_spot;
    double survival = 1.0;

    for (int64_t step = 0; step < c.steps; ++step) {
        const double previous = x;
        x += c.drift_step + c.vol_step * (sign * normals[static_cast<size_t>(step)]);

        switch (c.instrument) {
        case PLUMBLINE_ASIAN:
            arithmetic_sum += std::exp(x);
            geometric_sum += x;
            break;

        case PLUMBLINE_LOOKBACK: {
            /* Exact bridge extremes. A path sampled only at the step ends
             * would miss every excursion between them, and no path count
             * removes that bias. */
            const double u = rng_uniforms.next_uniform();
            const double gap = x - previous;
            const double span = std::sqrt(gap * gap - 2.0 * c.var_step * std::log(u));
            const double high = 0.5 * (previous + x + span);
            const double low = 0.5 * (previous + x - span);
            if (high > running_max) {
                running_max = high;
            }
            if (low < running_min) {
                running_min = low;
            }
            break;
        }

        case PLUMBLINE_BARRIER: {
            /* Once survival reaches zero it stays there, so the remaining
             * steps only have to carry the path to expiry for the control. */
            if (survival > 0.0) {
                const double gap_previous = previous - c.log_barrier;
                const double gap_now = x - c.log_barrier;
                const bool breached =
                    c.barrier_is_down ? (gap_now <= 0.0) : (gap_now >= 0.0);
                if (breached) {
                    survival = 0.0;
                } else {
                    /* Bridge no-hit probability across the step just taken.
                     * Both gaps are positive here: a non-positive previous gap
                     * would mean the path was already dead. */
                    const double product = gap_previous * gap_now;
                    survival *= 1.0 - std::exp(-2.0 * product / c.var_step);
                }
            }
            break;
        }

        default:
            break;
        }
    }

    const double terminal = std::exp(x);
    const double vanilla = positive_part(c.phi * (terminal - c.strike));

    switch (c.instrument) {
    case PLUMBLINE_ASIAN: {
        const double steps = static_cast<double>(c.steps);
        const double geometric_average = std::exp(geometric_sum / steps);
        const double average =
            c.averaging_arithmetic ? (arithmetic_sum / steps) : geometric_average;
        payoff = c.discount * positive_part(c.phi * (average - c.strike));
        control = c.discount * positive_part(c.phi * (geometric_average - c.strike));
        break;
    }

    case PLUMBLINE_LOOKBACK: {
        if (!c.strike_fixed) {
            payoff = c.phi > 0.0 ? c.discount * (terminal - std::exp(running_min))
                                 : c.discount * (std::exp(running_max) - terminal);
        } else {
            const double extreme =
                c.phi > 0.0 ? std::exp(running_max) : std::exp(running_min);
            payoff = c.discount * positive_part(c.phi * (extreme - c.strike));
        }
        control = c.discount * vanilla;
        break;
    }

    case PLUMBLINE_BARRIER: {
        const double weight = c.barrier_is_out ? survival : (1.0 - survival);
        payoff = c.discount * vanilla * weight;
        control = c.discount * vanilla;
        break;
    }

    default:
        payoff = 0.0;
        control = 0.0;
        break;
    }
}

bool needs_path(int64_t instrument) {
    return instrument == PLUMBLINE_ASIAN || instrument == PLUMBLINE_BARRIER ||
           instrument == PLUMBLINE_LOOKBACK;
}

/* One RNG block: a fixed number of path-pairs drawn from one stream. */
void run_block(const Contract &c, uint64_t seed, int64_t block_index,
               int64_t pairs, bool antithetic, Accumulator &accumulator,
               std::vector<double> &normal_scratch) {
    Xoshiro256pp normals_rng(seed, static_cast<uint64_t>(block_index));
    /* Uniforms for the lookback bridge come from a stream of their own, so
     * adding or removing that draw cannot shift the normals. */
    Xoshiro256pp uniforms_rng(seed ^ 0xA5A5A5A5A5A5A5A5ULL,
                              static_cast<uint64_t>(block_index));

    const bool stepped = needs_path(c.instrument);

    for (int64_t pair = 0; pair < pairs; ++pair) {
        double payoff_up = 0.0, control_up = 0.0;
        double payoff_down = 0.0, control_down = 0.0;

        if (!stepped) {
            const double z = normals_rng.next_normal();
            terminal_sample(c, z, payoff_up, control_up);
            if (antithetic) {
                terminal_sample(c, -z, payoff_down, control_down);
            }
        } else {
            for (int64_t step = 0; step < c.steps; ++step) {
                normal_scratch[static_cast<size_t>(step)] = normals_rng.next_normal();
            }
            stepped_sample(c, uniforms_rng, 1.0, normal_scratch, payoff_up, control_up);
            if (antithetic) {
                stepped_sample(c, uniforms_rng, -1.0, normal_scratch, payoff_down,
                               control_down);
            }
        }

        if (antithetic) {
            accumulator.push(0.5 * (payoff_up + payoff_down),
                             0.5 * (control_up + control_down));
        } else {
            accumulator.push(payoff_up, control_up);
        }
    }
}

}  // namespace

/* ------------------------------------------------------------------------ */

extern "C" {

PLUMBLINE_API const char *plumbline_backend_version(void) { return PLUMBLINE_BACKEND_VERSION; }

PLUMBLINE_API int64_t plumbline_backend_threads(void) {
    const unsigned int found = std::thread::hardware_concurrency();
    return found > 0 ? static_cast<int64_t>(found) : 1;
}

PLUMBLINE_API int64_t plumbline_request_size(void) {
    return static_cast<int64_t>(sizeof(PlumblineMCRequest));
}

PLUMBLINE_API int64_t plumbline_result_size(void) {
    return static_cast<int64_t>(sizeof(PlumblineMCResult));
}

PLUMBLINE_API int32_t plumbline_mc_price(const PlumblineMCRequest *request,
                           PlumblineMCResult *result) {
    if (request == nullptr || result == nullptr) {
        return PLUMBLINE_BAD_PARAMETER;
    }
    if (request->struct_size != static_cast<int64_t>(sizeof(PlumblineMCRequest)) ||
        result->struct_size != static_cast<int64_t>(sizeof(PlumblineMCResult))) {
        return PLUMBLINE_BAD_STRUCT_SIZE;
    }
    if (request->instrument < PLUMBLINE_EUROPEAN ||
        request->instrument > PLUMBLINE_LOOKBACK) {
        return PLUMBLINE_BAD_INSTRUMENT;
    }
    if (request->paths <= 0 || request->spot <= 0.0 || request->strike < 0.0) {
        return PLUMBLINE_BAD_PARAMETER;
    }
    /* A degenerate contract has no randomness in it, and the caller owns the
     * exact closed form for that case. Refuse rather than return a number the
     * caller would then have to second-guess. */
    if (request->maturity <= 0.0 || request->volatility <= 0.0) {
        return PLUMBLINE_DEGENERATE_INPUT;
    }
    if (needs_path(request->instrument) && request->steps <= 0) {
        return PLUMBLINE_BAD_PARAMETER;
    }
    if (request->instrument == PLUMBLINE_BARRIER && request->barrier <= 0.0) {
        return PLUMBLINE_BAD_PARAMETER;
    }

    Contract contract{};
    contract.instrument = request->instrument;
    contract.phi = request->is_call ? 1.0 : -1.0;
    contract.spot = request->spot;
    contract.strike = request->strike;
    contract.maturity = request->maturity;
    contract.discount = std::exp(-request->rate * request->maturity);

    const double variance = request->volatility * request->volatility;
    contract.drift_total =
        (request->rate - request->dividend - 0.5 * variance) * request->maturity;
    contract.vol_total = request->volatility * std::sqrt(request->maturity);

    contract.steps = request->steps;
    if (needs_path(request->instrument)) {
        const double dt = request->maturity / static_cast<double>(request->steps);
        contract.drift_step = (request->rate - request->dividend - 0.5 * variance) * dt;
        contract.vol_step = request->volatility * std::sqrt(dt);
        contract.var_step = variance * dt;
    }

    contract.log_barrier =
        request->barrier > 0.0 ? std::log(request->barrier) : 0.0;
    contract.barrier_is_down = request->barrier_is_down != 0;
    contract.barrier_is_out = request->barrier_is_out != 0;
    contract.averaging_arithmetic = request->averaging_arithmetic != 0;
    contract.payout_asset = request->payout_asset != 0;
    contract.cash_amount = request->cash_amount;
    contract.strike_fixed = request->strike_fixed != 0;

    const bool antithetic = request->antithetic != 0;
    const int64_t total_pairs =
        antithetic ? (request->paths / 2 > 0 ? request->paths / 2 : 1) : request->paths;
    const int64_t block_pairs =
        request->block_pairs > 0 ? request->block_pairs : kDefaultBlockPairs;
    const int64_t blocks = (total_pairs + block_pairs - 1) / block_pairs;

    int64_t threads = request->threads;
    if (threads <= 0) {
        threads = plumbline_backend_threads();
    }
    if (threads > blocks) {
        threads = blocks;
    }
    if (threads < 1) {
        threads = 1;
    }

    /* One accumulator per block, not per thread.
     *
     * Merging Welford accumulators is not associative in floating point, so if
     * each thread merged whatever blocks it happened to win from the atomic
     * counter, the last bits of the answer would depend on the scheduler. Two
     * runs on the same machine with the same seed would disagree, which for a
     * tool that judges other people's numerics is not a rounding detail.
     *
     * Each block therefore gets its own slot, written by exactly one thread,
     * and the merge afterwards walks the slots in index order. The result is
     * identical on one thread and on all of them, run after run. */
    std::vector<Accumulator> block_results(static_cast<size_t>(blocks));
    std::atomic<int64_t> next_block{0};

    auto worker = [&]() {
        std::vector<double> scratch(
            static_cast<size_t>(needs_path(contract.instrument) ? contract.steps : 1));
        for (;;) {
            const int64_t block = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block >= blocks) {
                break;
            }
            const int64_t start = block * block_pairs;
            const int64_t pairs =
                (start + block_pairs <= total_pairs) ? block_pairs : (total_pairs - start);
            if (pairs <= 0) {
                continue;
            }
            Accumulator local;
            run_block(contract, request->seed, block, pairs, antithetic, local, scratch);
            block_results[static_cast<size_t>(block)] = local;
        }
    };

    if (threads == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(threads));
        for (int64_t index = 0; index < threads; ++index) {
            pool.emplace_back(worker);
        }
        for (auto &thread : pool) {
            thread.join();
        }
    }

    Accumulator total;
    for (const Accumulator &block_result : block_results) {
        total.merge(block_result);
    }

    const int64_t n = total.count;
    double price = total.mean_y;
    double beta = 0.0;
    double variance_of_estimator = n > 1 ? total.m2_y / static_cast<double>(n - 1) : 0.0;

    if (request->control_variate != 0 && n > 1) {
        const double var_x = total.m2_x / static_cast<double>(n - 1);
        const double cov = total.comoment / static_cast<double>(n - 1);
        if (var_x > 1e-16) {
            beta = cov / var_x;
            price = total.mean_y - beta * (total.mean_x - request->control_mean);
            const double var_y = total.m2_y / static_cast<double>(n - 1);
            variance_of_estimator = var_y - 2.0 * beta * cov + beta * beta * var_x;
            if (variance_of_estimator < 0.0) {
                variance_of_estimator = 0.0;
            }
        }
    }

    result->price = price;
    result->standard_error =
        n > 1 ? std::sqrt(variance_of_estimator / static_cast<double>(n)) : 0.0;
    result->control_beta = beta;
    result->paths = antithetic ? 2 * n : n;
    result->threads = threads;
    result->blocks = blocks;
    return PLUMBLINE_OK;
}

}  // extern "C"
