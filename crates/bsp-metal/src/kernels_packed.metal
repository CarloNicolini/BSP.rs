#include <metal_stdlib>
using namespace metal;
struct Config {
  uint n, m;
  float epsilon, damping;
  uint observables, width;
};
struct Control {
  atomic_uint error;
  uint done, sweeps, paramagnetic;
  float residual, complexity;
  uint unused[2];
};
inline bool stopped(device Control &s) {
  return s.done || atomic_load_explicit(&s.error, memory_order_relaxed);
}
inline float log_one_plus(float x) {
  float y = 1 + x;
  return y == 1 ? x : log(y) * (x / (y - 1));
}
inline float checked_product(float a, float b, device Control &s) {
  float p = a * b;
  if (a > 0 && b > 0 && p == 0)
    atomic_fetch_or_explicit(&s.error, 4u, memory_order_relaxed);
  return p;
}
// Messages cross between clause-major and variable-major order without random
// DRAM traffic. Slot k of a packed buffer belongs to one bucket; a bucket holds
// the edges of one source window, ordered as their consumer visits them.
// pack_eta gathers eta inside one edge window per bucket; cavities read
// eta_packed[eta_at[p]], and write q to q_packed[q_slot[p]] inside one position
// window per bucket; clause kernels read q_packed[q_at[e]].
// Observables flag edges of inactive clauses with the eta sign bit.
inline float eta_of(float x) { return abs(x); }
inline bool active_of(float x) { return !(as_type<uint>(x) >> 31); }
// Cavities mark incidences of fixed variables with a q outside [0,1].
constant float FIXED = -1;
inline bool free_q(float q) { return !(q < 0); }
// Observables carry each normalizer with its q to clause_observables.
inline void put_q(device float *q_packed, uint k, float q, float z,
                  constant Config &cfg) {
  if (cfg.observables)
    ((device float2 *)q_packed)[k] = float2(q, z);
  else
    q_packed[k] = q;
}
inline uint flagged_eta(uint e, const device float *eta, const device uint *ec,
                        const device uint *active, constant Config &cfg) {
  uint x = as_type<uint>(eta[e]) & 0x7fffffffu;
  return cfg.observables && !active[ec[e]] ? x | 0x80000000u : x;
}
// Each thread packs four slots; the edge count need not be a multiple of four.
kernel void pack_eta(const device uint *ec [[buffer(4)]],
                     const device uint *active [[buffer(6)]],
                     const device float *eta [[buffer(8)]],
                     const device uint *eta_source [[buffer(11)]],
                     device float *eta_packed [[buffer(30)]],
                     device Control &state [[buffer(19)]],
                     constant Config &cfg [[buffer(20)]],
                     constant uint &edges [[buffer(26)]],
                     uint t [[thread_position_in_grid]]) {
  uint k = 4 * t;
  if (k >= edges || state.done)
    return;
  if (k + 4 <= edges) {
    uint4 e = ((const device uint4 *)eta_source)[t];
    ((device uint4 *)eta_packed)[t] =
        uint4(flagged_eta(e.x, eta, ec, active, cfg),
              flagged_eta(e.y, eta, ec, active, cfg),
              flagged_eta(e.z, eta, ec, active, cfg),
              flagged_eta(e.w, eta, ec, active, cfg));
  } else
    for (; k < edges; ++k)
      eta_packed[k] =
          as_type<float>(flagged_eta(eta_source[k], eta, ec, active, cfg));
}
#define CAVITY_BUFFERS                                                         \
  const device uint *vo [[buffer(0)]],                                         \
      const device char *assignment [[buffer(5)]],                             \
      device uint *degree [[buffer(7)]],                                       \
      const device float *eta_packed [[buffer(30)]],                           \
      const device uint *eta_at [[buffer(27)]],                                \
      device float *q [[buffer(10)]], device float *q_packed [[buffer(1)]],    \
      const device uint *q_slot [[buffer(28)]],                                \
      device float2 *products [[buffer(13)]],                                  \
      device Control &state [[buffer(19)]],                                    \
      constant Config &cfg [[buffer(20)]]
kernel void cavities(CAVITY_BUFFERS, const device uint *ids [[buffer(21)]],
                     uint id [[thread_position_in_grid]]) {
  if (id >= ids[0])
    return;
  uint v = ids[id + 1];
  if (v >= cfg.n || stopped(state))
    return;
  uint original_degree = vo[2 * v + 2] - vo[2 * v];
  if (original_degree <= 32 || original_degree >= 128)
    return;
  if (assignment[v] >= 0) {
    for (uint i = vo[2 * v]; i < vo[2 * v + 2]; ++i)
      put_q(q_packed, q_slot[i], FIXED, 0, cfg);
    return;
  }
  float p[2] = {1, 1};
  uint d = 0;
  for (uint sign = 0; sign < 2; ++sign) {
    uint g = 2 * v + sign;
    for (uint i = vo[g]; i < vo[g + 1]; ++i) {
      float x = eta_packed[eta_at[i]];
      q[i] = p[sign];
      p[sign] = checked_product(p[sign], 1 - eta_of(x), state);
      if (cfg.observables)
        d += active_of(x);
    }
  }
  if (cfg.observables) {
    products[v] = float2(p[0], p[1]);
    degree[v] = d;
  }
  for (uint sign = 0; sign < 2; ++sign) {
    uint g = 2 * v + sign;
    float suffix = 1, opposite = p[1 - sign];
    for (uint i = vo[g + 1]; i > vo[g];) {
      --i;
      float x = eta_packed[eta_at[i]];
      float same = checked_product(q[i], suffix, state);
      suffix = checked_product(suffix, 1 - eta_of(x), state);
      float z = opposite + (1 - opposite) * same;
      if (!(z > 0) || !isfinite(z)) {
        put_q(q_packed, q_slot[i], q[i], z, cfg);
        atomic_fetch_or_explicit(&state.error, 1u, memory_order_relaxed);
        return;
      }
      q[i] = clamp((1 - opposite) * same / z, 0.0f, 1.0f);
      put_q(q_packed, q_slot[i], q[i], z, cfg);
    }
  }
}

kernel void cavities_simd(CAVITY_BUFFERS, uint i [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  uint v = i / 32;
  if (v >= cfg.n || state.done)
    return;
  uint start = vo[2 * v], split = vo[2 * v + 1], end = vo[2 * v + 2];
  if (end - start > 32)
    return;
  if (assignment[v] >= 0) {
    if (start + lane < end)
      put_q(q_packed, q_slot[start + lane], FIXED, 0, cfg);
    return;
  }
  uint p = start + lane;
  bool valid = p < end;
  float x = valid ? eta_packed[eta_at[p]] : 0;
  float factor = valid ? 1 - eta_of(x) : 1;
  float fp = valid && p < split ? factor : 1,
        fm = valid && p >= split ? factor : 1;
  // Suffix scans run on lane-reversed factors (lane^31 == 31-lane); one
  // shuffle returns both signs with the association of separate scans.
  uint r = start + 31 - lane;
  float rf = simd_shuffle_xor(factor, 31);
  float rp = r < split ? rf : 1, rm = r < end && r >= split ? rf : 1;
  float plus = simd_product(fp), minus = simd_product(fm);
  // Underflow tests and zero counts run only when some lane can need them.
  if ((plus == 0 || minus == 0) &&
      ((plus == 0 && simd_all(fp > 0)) || (minus == 0 && simd_all(fm > 0))))
    atomic_fetch_or_explicit(&state.error, 4u, memory_order_relaxed);
  float pp = simd_prefix_exclusive_product(fp),
        pm = simd_prefix_exclusive_product(fm);
  float sp = simd_prefix_exclusive_product(rp),
        sm = simd_prefix_exclusive_product(rm);
  float suffix = simd_shuffle_xor(r < split ? sp : sm, 31);
  uint d = cfg.observables ? simd_sum(valid ? uint(active_of(x)) : 0u) : 0;
  if (!lane && cfg.observables) {
    degree[v] = d;
    products[v] = float2(plus, minus);
  }
  float same = p < split ? pp * suffix : pm * suffix;
  bool zero = valid && factor == 0 && same == 0;
  if (simd_any(zero)) {
    uint zero_plus = simd_sum(uint(fp == 0)),
         zero_minus = simd_sum(uint(fm == 0));
    if (zero && (p < split ? zero_plus : zero_minus) == 1)
      atomic_fetch_or_explicit(&state.error, 4u, memory_order_relaxed);
  }
  if (valid) {
    float opposite = p < split ? minus : plus;
    float z = opposite + (1 - opposite) * same;
    if (!(z > 0) || !isfinite(z))
      atomic_fetch_or_explicit(&state.error, 1u, memory_order_relaxed);
    else
      put_q(q_packed, q_slot[p], clamp((1 - opposite) * same / z, 0.0f, 1.0f),
            z, cfg);
  }
}

// High-degree variables use one cooperative group. Scans retain exclusion
// products for zero factors; reassociation belongs to the opt-in f32 contract.
kernel void cavities_wide(CAVITY_BUFFERS, const device uint *ids [[buffer(24)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]]) {
  if (group >= ids[0])
    return;
  uint v = ids[group + 1];
  // Only uniform conditions may bypass a group containing barriers.
  if (state.done)
    return;
  if (assignment[v] >= 0) {
    for (uint i = vo[2 * v] + lane; i < vo[2 * v + 2]; i += 256)
      put_q(q_packed, q_slot[i], FIXED, 0, cfg);
    return;
  }
  threadgroup float temp[256];
  threadgroup uint counts[256];
  float p[2] = {1, 1};
  uint d = 0;
  for (uint sign = 0; sign < 2; ++sign) {
    uint start = vo[2 * v + sign], end = vo[2 * v + sign + 1];
    float carry = 1;
    for (uint base = start; base < end; base += 256) {
      uint i = base + lane;
      float x = i < end ? eta_packed[eta_at[i]] : 0,
            factor = i < end ? 1 - eta_of(x) : 1;
      if (i < end && cfg.observables)
        d += active_of(x);
      temp[lane] = factor;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint step = 1; step < 256; step <<= 1) {
        float earlier = lane >= step ? temp[lane - step] : 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        temp[lane] = checked_product(temp[lane], earlier, state);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (i < end)
        q[i] = checked_product(carry, lane ? temp[lane - 1] : 1, state);
      carry = checked_product(carry, temp[255], state);
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    p[sign] = carry;
  }
  counts[lane] = d;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 128; step; step >>= 1) {
    if (lane < step)
      counts[lane] += counts[lane + step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane && cfg.observables) {
    degree[v] = counts[0];
    products[v] = float2(p[0], p[1]);
  }
  for (uint sign = 0; sign < 2; ++sign) {
    uint start = vo[2 * v + sign], end = vo[2 * v + sign + 1];
    float carry = 1, opposite = p[1 - sign];
    for (uint done = 0; done < end - start; done += 256) {
      bool valid = done + lane < end - start;
      uint i = valid ? end - 1 - done - lane : start;
      float x = valid ? eta_packed[eta_at[i]] : 0;
      temp[lane] = valid ? 1 - eta_of(x) : 1;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint step = 1; step < 256; step <<= 1) {
        float earlier = lane >= step ? temp[lane - step] : 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        temp[lane] = checked_product(temp[lane], earlier, state);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (valid) {
        float suffix = checked_product(carry, lane ? temp[lane - 1] : 1, state);
        float same = checked_product(q[i], suffix, state),
              z = opposite + (1 - opposite) * same;
        if (!(z > 0) || !isfinite(z)) {
          put_q(q_packed, q_slot[i], q[i], z, cfg);
          atomic_fetch_or_explicit(&state.error, 1u, memory_order_relaxed);
        } else {
          q[i] = clamp((1 - opposite) * same / z, 0.0f, 1.0f);
          put_q(q_packed, q_slot[i], q[i], z, cfg);
        }
      }
      carry = checked_product(carry, temp[255], state);
      threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
  }
}

struct FeistelConfig {
  ulong sign_key;
  uint half_bits, fluctuating, repairs, unused;
};

// SplitMix64 finalizer and reverse cycle walk, matching the Rust generator.
inline ulong feistel_mix(ulong x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
  return x ^ (x >> 31);
}
inline uint inverse_matching(uint x, uint limit, uint bits,
                             const device ulong *keys) {
  uint mask = (1u << bits) - 1;
  do {
    uint left = x >> bits, right = x & mask;
    for (int round = 7; round >= 0; --round) {
      uint previous =
          right ^ (uint(feistel_mix(ulong(left) ^ keys[round])) & mask);
      right = left;
      left = previous;
    }
    x = (left << bits) | right;
  } while (x >= limit);
  return x;
}
inline uint lower_bound_variable(const device uint2 *rows, uint count,
                                 uint variable) {
  uint first = 0;
  while (first < count) {
    uint mid = first + (count - first) / 2;
    if (rows[mid].x < variable)
      first = mid + 1;
    else
      count = mid;
  }
  return first;
}
inline bool repaired(const device uint *clauses, uint count, uint clause) {
  uint first = 0, last = count;
  while (first < last) {
    uint mid = first + (last - first) / 2;
    uint candidate = clauses[mid];
    if (candidate == clause)
      return true;
    if (candidate < clause)
      first = mid + 1;
    else
      last = mid;
  }
  return false;
}

// One owner per variable writes q directly to clause-major slots. Only the
// original stub intervals and sparse repair exceptions need device metadata.
kernel void cavities_feistel(const device uint *stub_offsets [[buffer(0)]],
                             device float *q [[buffer(1)]],
                             const device uint *repairs [[buffer(2)]],
                             const device uint2 *insertions [[buffer(4)]],
                             const device char *assignment [[buffer(5)]],
                             const device uint *active [[buffer(6)]],
                             device uint *degree [[buffer(7)]],
                             const device float *eta [[buffer(8)]],
                             const device ulong *keys [[buffer(11)]],
                             device float2 *products [[buffer(13)]],
                             device Control &state [[buffer(19)]],
                             constant Config &cfg [[buffer(20)]],
                             constant FeistelConfig &f [[buffer(21)]],
                             uint v [[thread_position_in_grid]]) {
  if (v >= cfg.n || stopped(state))
    return;
  uint k = cfg.width, group = 0, start, count, inserted = 0, added = 0,
       limit = cfg.m;
  if (f.fluctuating) {
    limit *= k;
    start = stub_offsets[v];
    count = stub_offsets[v + 1] - start;
    inserted = lower_bound_variable(insertions, f.repairs * k, v);
    added = lower_bound_variable(insertions, f.repairs * k, v + 1) - inserted;
  } else {
    uint base = cfg.n / k, extra = cfg.n % k, split = extra * (base + 1);
    group = v < split ? v / (base + 1) : extra + (v - split) / base;
    uint local = v - (group * base + min(group, extra)),
         size = base + uint(group < extra);
    uint d = cfg.m / size, remainder = cfg.m % size;
    start = local * d + min(local, remainder);
    count = d + uint(local < remainder);
  }
  float p[2] = {1, 1}, suffix[2] = {1, 1};
  uint d = 0, total = count + added;
  bool fixed = assignment[v] >= 0;
  for (uint pass = 0; pass < (fixed ? 1u : 2u); ++pass) {
    // Reverse the traversal for suffixes. Prefixes live in the output
    // slots until overwritten with the full zero-safe exclusion product.
    for (uint i = 0; i < total; ++i) {
      uint pos = pass ? total - 1 - i : i;
      uint e;
      if (pos < count) {
        e = inverse_matching(start + pos, limit, f.half_bits, keys + 8 * group);
        if (f.fluctuating) {
          if (repaired(repairs, f.repairs, e / k))
            continue;
        } else
          e = e * k + group;
      } else
        e = insertions[inserted + pos - count].y;
      if (!active[e / k])
        continue;
      if (fixed) {
        put_q(q, e, FIXED, 0, cfg);
        continue;
      }
      uint sign = uint(feistel_mix(ulong(e) ^ f.sign_key)) & 1u;
      float factor = 1 - eta[e];
      if (!pass) {
        put_q(q, e, p[sign], 0, cfg);
        p[sign] = checked_product(p[sign], factor, state);
        ++d;
      } else {
        float prefix = cfg.observables ? ((device float2 *)q)[e].x : q[e];
        float same = checked_product(prefix, suffix[sign], state);
        suffix[sign] = checked_product(suffix[sign], factor, state);
        float opposite = p[1 - sign], z = opposite + (1 - opposite) * same;
        if (!(z > 0) || !isfinite(z)) {
          atomic_fetch_or_explicit(&state.error, 1u, memory_order_relaxed);
          return;
        }
        put_q(q, e, clamp((1 - opposite) * same / z, 0.0f, 1.0f), z, cfg);
      }
    }
  }
  if (cfg.observables && !fixed) {
    products[v] = float2(p[0], p[1]);
    degree[v] = d;
  }
}

// Short clauses retain prefixes in registers; arbitrary widths use E-sized
// scratch.
template <uint Width, bool Implicit>
inline void clause_update(uint c, uint start, uint end,
                          const device uint *active, device float *eta,
                          const device float *q_packed, const device uint *q_at,
                          device float *scratch, device float *residual,
                          device Control &state, constant Config &cfg) {
  if (stopped(state))
    return;
  if (!active[c]) {
    residual[c] = 0;
    return;
  }
  uint free_count = 0;
  float product = 1, local[Width ? Width : 1], local_q[Width ? Width : 1];
  for (uint e = start; e < end; ++e) {
    float value = q_packed[Implicit ? e : q_at[e]];
    if (Width)
      local_q[e - start] = value;
    if (!free_q(value))
      continue;
    if (Width)
      local[e - start] = product;
    else
      scratch[e] = product;
    product *= value;
    ++free_count;
  }
  if (!free_count) {
    atomic_fetch_or_explicit(&state.error, 2u, memory_order_relaxed);
    return;
  }
  float suffix = 1, r = 0;
  for (uint e = end; e > start;) {
    --e;
    float value = Width ? local_q[e - start] : q_packed[Implicit ? e : q_at[e]];
    if (!free_q(value))
      continue;
    float raw = (Width ? local[e - start] : scratch[e]) * suffix;
    suffix *= value;
    if (raw < 1e-16f)
      raw = 0;
    r = max(r, abs(raw - eta[e]));
    eta[e] = raw + (eta[e] - raw) * cfg.damping;
  }
  residual[c] = r;
}
#define CLAUSE_KERNEL(Name, Width)                                             \
  kernel void Name(const device uint *offsets [[buffer(2)]],                   \
                   const device uint *active [[buffer(6)]],                    \
                   device float *eta [[buffer(8)]],                            \
                   device float *q_packed [[buffer(1)]],                       \
                   const device uint *q_at [[buffer(29)]],                     \
                   device float *scratch [[buffer(12)]],                       \
                   device float *residual [[buffer(15)]],                      \
                   device Control &state [[buffer(19)]],                       \
                   constant Config &cfg [[buffer(20)]],                        \
                   const device uint *ids [[buffer(25)]],                      \
                   constant uint &base [[buffer(26)]],                         \
                   uint i [[thread_position_in_grid]]) {                       \
    if (cfg.width) {                                                           \
      if (i >= cfg.m)                                                          \
        return;                                                                \
      /* The second half of q is available as scratch during sweeps. */        \
      clause_update<Width, true>(i, i * cfg.width, (i + 1) * cfg.width,        \
                                 active, eta, q_packed, q_at, q_packed + base, \
                                 residual, state, cfg);                        \
    } else {                                                                   \
      if (i >= ids[base])                                                      \
        return;                                                                \
      uint c = ids[base + 1 + i];                                              \
      clause_update<Width, false>(c, offsets[c], offsets[c + 1], active, eta,  \
                                  q_packed, q_at, scratch, residual, state,    \
                                  cfg);                                        \
    }                                                                          \
  }
CLAUSE_KERNEL(clauses3, 3)
CLAUSE_KERNEL(clauses4, 4)
CLAUSE_KERNEL(clauses6, 6)
CLAUSE_KERNEL(clauses12, 12)
CLAUSE_KERNEL(clauses_general, 0)

kernel void residual_blocks(const device float *r [[buffer(15)]],
                            device Control &s [[buffer(19)]],
                            constant Config &cfg [[buffer(20)]],
                            device float4 *blocks [[buffer(23)]],
                            uint i [[thread_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]],
                            uint group [[threadgroup_position_in_grid]]) {
  if (stopped(s))
    return;
  threadgroup float temp[256];
  temp[lane] = i < cfg.m ? r[i] : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride; stride >>= 1) {
    if (lane < stride)
      temp[lane] = max(temp[lane], temp[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane)
    blocks[group].x = temp[0];
}
kernel void residual_final(device Control &s [[buffer(19)]],
                           constant Config &cfg [[buffer(20)]],
                           const device float4 *blocks [[buffer(23)]],
                           uint lane [[thread_index_in_threadgroup]]) {
  if (stopped(s))
    return;
  float r = 0;
  for (uint i = lane; i < (cfg.m + 255) / 256; i += 256)
    r = max(r, blocks[i].x);
  threadgroup float temp[256];
  temp[lane] = r;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride; stride >>= 1) {
    if (lane < stride)
      temp[lane] = max(temp[lane], temp[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane) {
    s.residual = temp[0];
    ++s.sweeps;
    s.done = temp[0] <= cfg.epsilon;
  }
}
kernel void variables(const device char *assignment [[buffer(5)]],
                      const device uint *degree [[buffer(7)]],
                      const device float2 *products [[buffer(13)]],
                      device float4 *surveys [[buffer(14)]],
                      device uint *paramagnetic [[buffer(16)]],
                      device float *terms [[buffer(17)]],
                      device Control &s [[buffer(19)]],
                      constant Config &cfg [[buffer(20)]],
                      uint v [[thread_position_in_grid]]) {
  if (v >= cfg.n)
    return;
  if (assignment[v] >= 0) {
    terms[v] = 0;
    paramagnetic[v] = 1;
    return;
  }
  float p = products[v].x, m = products[v].y, z = p + (1 - p) * m;
  if (!(z > 0) || !isfinite(z)) {
    atomic_fetch_or_explicit(&s.error, 1u, memory_order_relaxed);
    return;
  }
  float st = (1 - p) * m / z, sf = (1 - m) * p / z;
  surveys[v] = float4(st, sf, clamp(p * m / z, 0.0f, 1.0f), 1 - min(st, sf));
  terms[v] = (float(degree[v]) - 1) * log(z);
  paramagnetic[v] = degree[v] < 2 || z == 1;
}
// Normalizers borrow saved_eta (slot 9) only for active/free incidences.
// Freeze replaces those scratch values; inactive/fixed payloads stay untouched.
kernel void clause_observables(const device uint *offsets [[buffer(2)]],
                               const device uint *active [[buffer(6)]],
                               const device float2 *q_packed [[buffer(1)]],
                               const device uint *q_at [[buffer(29)]],
                               device float *norms [[buffer(9)]],
                               device float *terms [[buffer(18)]],
                               device Control &s [[buffer(19)]],
                               constant Config &cfg [[buffer(20)]],
                               uint c [[thread_position_in_grid]]) {
  if (c >= cfg.m)
    return;
  if (!active[c]) {
    terms[c] = 0;
    return;
  }
  float product = 1, sum = 0, correction = 0;
  uint start = cfg.width ? c * cfg.width : offsets[c],
       end = cfg.width ? (c + 1) * cfg.width : offsets[c + 1];
  for (uint e = start; e < end; ++e) {
    float2 qz = q_packed[cfg.width ? e : q_at[e]];
    if (!free_q(qz.x))
      continue;
    if (qz.y > 0 && isfinite(qz.y))
      norms[e] = qz.y;
    else
      qz.y = norms[e];
    product *= qz.x;
    float y = log(qz.y) - correction, t = sum + y;
    correction = (t - sum) - y;
    sum = t;
  }
  if (product >= 1) {
    atomic_fetch_or_explicit(&s.error, 1u, memory_order_relaxed);
    return;
  }
  terms[c] = sum + log_one_plus(-product);
}
kernel void observable_blocks(const device float *vt [[buffer(17)]],
                              const device float *ct [[buffer(18)]],
                              const device uint *paramagnetic [[buffer(16)]],
                              constant Config &cfg [[buffer(20)]],
                              device float4 *blocks [[buffer(23)]],
                              uint i [[thread_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
  threadgroup float2 temp[256];
  threadgroup uint flags[256];
  temp[lane] = float2(i < cfg.n ? vt[i] : 0, i < cfg.m ? ct[i] : 0);
  flags[lane] = i < cfg.n ? paramagnetic[i] : 1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride; stride >>= 1) {
    if (lane < stride) {
      temp[lane] += temp[lane + stride];
      flags[lane] &= flags[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane)
    blocks[group] = float4(temp[0], float(flags[0]), 0);
}
kernel void observable_final(device Control &s [[buffer(19)]],
                             constant Config &cfg [[buffer(20)]],
                             const device float4 *blocks [[buffer(23)]],
                             uint i [[thread_position_in_grid]]) {
  if (i)
    return;
  float2 sum = 0, correction = 0;
  uint paramagnetic = 1;
  for (uint j = 0; j < (max(cfg.n, cfg.m) + 255) / 256; ++j) {
    float2 y = blocks[j].xy - correction, t = sum + y;
    correction = (t - sum) - y;
    sum = t;
    paramagnetic &= blocks[j].z != 0;
  }
  s.paramagnetic = paramagnetic;
  s.complexity = paramagnetic ? 0 : sum.y - sum.x;
  if (!isfinite(s.complexity))
    atomic_fetch_or_explicit(&s.error, 1u, memory_order_relaxed);
}
