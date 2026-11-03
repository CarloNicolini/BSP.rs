#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <new>

struct Control {
  unsigned error, done, sweeps, paramagnetic;
  double residual, complexity;
};
struct Patch {
  unsigned index, op;
  int value;
};
constexpr unsigned SLOTS = 36;
struct Params {
  void *b[SLOTS];
  unsigned n, m, base, count, observables;
  double epsilon, damping;
};
template <class T> struct Pair {
  T x, y;
};
template <class T> struct Quad {
  T x, y, z, w;
};
template <class T> __device__ T *buf(const Params &p, int i) {
  return static_cast<T *>(p.b[i]);
}
__device__ unsigned tid() { return blockIdx.x * blockDim.x + threadIdx.x; }
template <class T> __device__ T maximum(T a, T b) { return a > b ? a : b; }
template <class T> __device__ T absolute(T a) { return a < 0 ? -a : a; }
template <class T> __device__ T clamp01(T a) {
  return a < 0 ? T(0) : (a > 1 ? T(1) : a);
}
__device__ float logarithm(float a) { return logf(a); }
__device__ double logarithm(double a) { return log(a); }
__device__ float log_one_plus(float a) { return log1pf(a); }
__device__ double log_one_plus(double a) { return log1p(a); }
// Error flags are consumed after kernel boundaries, so their writers need no
// polling.
__device__ bool stopped(Control &s) { return s.done; }
template <class T> __device__ T product(T a, T b, Control &s) {
  T c = a * b;
  if (a > 0 && b > 0 && c == 0)
    atomicOr(&s.error, 4u);
  return c;
}
template <class T> __device__ T inclusive_scan(T a) {
  unsigned lane = threadIdx.x & 31;
  for (unsigned d = 1; d < 32; d *= 2) {
    T b = __shfl_up_sync(0xffffffffu, a, d);
    if (lane >= d)
      a *= b;
  }
  return a;
}
template <class T> __device__ T suffix_scan(T a) {
  unsigned lane = threadIdx.x & 31;
  for (unsigned d = 1; d < 32; d *= 2) {
    T b = __shfl_down_sync(0xffffffffu, a, d);
    if (lane + d < 32)
      a *= b;
  }
  return a;
}
template <class T> __device__ T exclusive_scan(T a) {
  T b = inclusive_scan(a);
  T c = __shfl_up_sync(0xffffffffu, b, 1);
  return (threadIdx.x & 31) ? c : T(1);
}

// Messages cross between clause order and variable order without random DRAM
// traffic. Each slot of a packed buffer belongs to one bucket; a bucket holds
// the edges of one source window in the order their consumer visits them.
// pack_eta gathers eta inside one edge window per bucket into eta_packed (32);
// cavities read eta_packed[eta_at[i]] (29) and write q to q_packed (1) at
// q_slot[i] (30) inside one position window per bucket; clause kernels read
// q_packed[q_at[e]] (31). Observables flag edges of inactive clauses with the
// eta sign bit, and carry each normalizer with its q as a pair.
template <class T> __device__ bool active_of(T x) { return !signbit(x); }
// Cavities mark incidences of fixed variables with a q outside [0, 1].
template <class T> __device__ bool free_q(T q) { return !(q < 0); }
// Read-only loads use the non-coherent path so that they can move past stores.
template <class T> __device__ T load(const Params &p, int slot, unsigned i) {
  return __ldg(buf<T>(p, slot) + i);
}
template <class T>
__device__ void put_q(const Params &p, unsigned i, T q, T z) {
  unsigned k = load<unsigned>(p, 30, i);
  if (p.observables)
    buf<Pair<T>>(p, 1)[k] = {q, z};
  else
    buf<T>(p, 1)[k] = q;
}
template <class T> __device__ T packed_eta(const Params &p, unsigned i) {
  return load<T>(p, 32, load<unsigned>(p, 29, i));
}
template <class T> __global__ void pack_eta(Params p) {
  unsigned k = tid();
  if (k >= p.count || buf<Control>(p, 19)->done)
    return;
  unsigned e = buf<unsigned>(p, 28)[k];
  T x = buf<T>(p, 8)[e];
  if (p.observables && !buf<unsigned>(p, 6)[buf<unsigned>(p, 4)[e]])
    x = -x;
  buf<T>(p, 32)[k] = x;
}
template <class T> __global__ void cavities(Params p) {
  unsigned *ids = buf<unsigned>(p, 21);
  unsigned id = tid();
  if (id >= ids[0])
    return;
  unsigned v = ids[id + 1];
  Control &s = *buf<Control>(p, 19);
  unsigned *vo = buf<unsigned>(p, 0), *degree = buf<unsigned>(p, 7);
  if (stopped(s))
    return;
  if (buf<signed char>(p, 5)[v] >= 0) {
    for (unsigned i = vo[2 * v]; i < vo[2 * v + 2]; ++i)
      put_q(p, i, T(-1), T(0));
    return;
  }
  T *q = buf<T>(p, 10);
  Pair<T> *products = buf<Pair<T>>(p, 13);
  T a[2] = {1, 1};
  unsigned d = 0;
  for (unsigned sign = 0; sign < 2; ++sign)
    for (unsigned i = vo[2 * v + sign]; i < vo[2 * v + sign + 1]; ++i) {
      T x = packed_eta<T>(p, i);
      q[i] = a[sign];
      a[sign] = product(a[sign], T(1) - absolute(x), s);
      if (p.observables)
        d += active_of(x);
    }
  if (p.observables) {
    products[v] = {a[0], a[1]};
    degree[v] = d;
  }
  for (unsigned sign = 0; sign < 2; ++sign) {
    T suffix = 1, opposite = a[1 - sign];
    for (unsigned i = vo[2 * v + sign + 1]; i > vo[2 * v + sign];) {
      --i;
      T same = product(q[i], suffix, s);
      suffix = product(suffix, T(1) - absolute(packed_eta<T>(p, i)), s);
      T z = opposite + (T(1) - opposite) * same;
      if (!(z > 0) || !isfinite(z)) {
        atomicOr(&s.error, 1u);
        return;
      }
      put_q(p, i, clamp01((T(1) - opposite) * same / z), z);
    }
  }
}
// One warp holds several variables of degree at most 32. Variable v takes a
// block of P lanes, P the power of two covering its degree, at a lane offset
// that is a multiple of P. Segmented scans inside the block multiply as the
// baseline warp does with the variable at lane 0 and unit padding, so every
// product, including the lane-31 totals, keeps its association.
// Slot 33 holds per warp {first variable, lanes starting a variable, lanes
// holding an edge, lanes holding a positive-literal edge}; slot 35 holds the
// position of the first edge. A start lane without an edge but with the
// positive bit marks a variable left to the wider kernels.
struct Seg {
  unsigned v, first, r, width, mask, i;
  bool mine, edge, positive, fixed, valid;
};
__device__ Seg prepare_seg(const Params &p, unsigned w, unsigned lane) {
  uint4 packed_warp = load<uint4>(p, 33, w);
  Quad<unsigned> warp = {packed_warp.x, packed_warp.y, packed_warp.z, packed_warp.w};
  unsigned position = load<unsigned>(p, 35, w);
  unsigned up_to = (2u << lane) - 1, below = warp.y & up_to;
  unsigned first = 31 - __clz(below), v = warp.x + __popc(below) - 1;
  unsigned after = warp.y & ~up_to;
  unsigned next = after ? __ffs(after) - 1 : 32;
  unsigned segment = (next == 32 ? ~0u : (1u << next) - 1) & ~((1u << first) - 1);
  unsigned degree = __popc(warp.z & segment), r = lane - first;
  bool wider = !(warp.z >> first & 1) && (warp.w >> first & 1);
  unsigned width = degree < 2 ? 1 : 1u << (32 - __clz(degree - 1));
  unsigned mask = (width == 32 ? ~0u : (1u << width) - 1) << first;
  bool mine = !wider && r < width, edge = warp.z >> lane & 1,
       positive = warp.w >> lane & 1;
  unsigned i = position + __popc(warp.z & (1u << lane) - 1);
  bool fixed = load<signed char>(p, 5, v) >= 0;
  return {v, first, r, width, mask, i, mine, edge, positive, fixed, edge && !fixed};
}
template <class T>
__device__ void finish_seg(const Params &p, Control &s, Seg g, T x, unsigned k) {
  T factor = g.valid ? T(1) - absolute(x) : T(1);
  T fp = g.valid && g.positive ? factor : T(1),
    fm = g.valid && !g.positive ? factor : T(1);
  T xp = fp, xm = fm, yp = fp, ym = fm;
  unsigned r = g.r, width = g.width;
  // A step at or above every lane's width does not multiply. Width 32 still
  // scans d = 16. Narrower records stop before that step.
  unsigned bound = __any_sync(0xffffffffu, width > 16u) ? 32u : 16u;
  for (unsigned d = 1; d < bound; d *= 2) {
    T a = __shfl_up_sync(0xffffffffu, xp, d), b = __shfl_up_sync(0xffffffffu, xm, d),
      c = __shfl_down_sync(0xffffffffu, yp, d), e = __shfl_down_sync(0xffffffffu, ym, d);
    if (r >= d) {
      xp *= a;
      xm *= b;
    }
    // A down-scan multiplies in the order of an up-scan over lane-reversed factors.
    if (r + d < width) {
      yp *= c;
      ym *= e;
    }
  }
  unsigned last = g.first + width - 1;
  T plus = __shfl_sync(0xffffffffu, xp, last), minus = __shfl_sync(0xffffffffu, xm, last);
  T pp = __shfl_up_sync(0xffffffffu, xp, 1), pm = __shfl_up_sync(0xffffffffu, xm, 1);
  T sp = __shfl_down_sync(0xffffffffu, yp, 1), sm = __shfl_down_sync(0xffffffffu, ym, 1);
  unsigned positive_plus = __ballot_sync(0xffffffffu, fp > 0) & g.mask,
           positive_minus = __ballot_sync(0xffffffffu, fm > 0) & g.mask,
           zero_plus = __popc(__ballot_sync(0xffffffffu, fp == 0) & g.mask),
           zero_minus = __popc(__ballot_sync(0xffffffffu, fm == 0) & g.mask),
           active = __popc(__ballot_sync(0xffffffffu, g.valid && active_of(x)) & g.mask);
  if (g.edge && g.fixed) {
    if (p.observables)
      buf<Pair<T>>(p, 1)[k] = {T(-1), T(0)};
    else
      buf<T>(p, 1)[k] = T(-1);
  }
  if (!g.mine || g.fixed)
    return;
  if (!r) {
    pp = pm = 1;
    if ((plus == 0 && positive_plus == g.mask) || (minus == 0 && positive_minus == g.mask))
      atomicOr(&s.error, 4u);
    if (p.observables) {
      buf<unsigned>(p, 7)[g.v] = active;
      buf<Pair<T>>(p, 13)[g.v] = {plus, minus};
    }
  }
  if (r == width - 1)
    sp = sm = 1;
  if (g.valid) {
    T same = g.positive ? pp * sp : pm * sm, opposite = g.positive ? minus : plus;
    // With exactly one zero, excluding that edge must produce a positive
    // product. The full-product check above cannot detect this underflow.
    if (factor == 0 && same == 0 && (g.positive ? zero_plus : zero_minus) == 1)
      atomicOr(&s.error, 4u);
    T z = opposite + (T(1) - opposite) * same;
    if (!(z > 0) || !isfinite(z))
      atomicOr(&s.error, 1u);
    else if (p.observables)
      buf<Pair<T>>(p, 1)[k] = {clamp01((T(1) - opposite) * same / z), z};
    else
      buf<T>(p, 1)[k] = clamp01((T(1) - opposite) * same / z);
  }
}
// One warp computes two packed records. Both gathers are issued before either
// scan, so a second miss can stay in flight while the first record is scanned.
template <class T> __global__ void cavities_simd(Params p) {
  unsigned lane = threadIdx.x & 31;
  unsigned records = p.count >> 5;
  unsigned workers = (records + 1u) >> 1;
  unsigned w = tid() >> 5;
  if (w >= workers)
    return;
  Control &s = *buf<Control>(p, 19);
  if (s.done)
    return;
  unsigned w0 = w << 1, w1 = w0 + 1;
  bool has1 = w1 < records;
  Seg a = prepare_seg(p, w0, lane);
  Seg b = {};
  if (has1)
    b = prepare_seg(p, w1, lane);
  unsigned at_a = a.valid ? load<unsigned>(p, 29, a.i) : 0;
  unsigned at_b = has1 && b.valid ? load<unsigned>(p, 29, b.i) : 0;
  unsigned slot_a = a.edge ? load<unsigned>(p, 30, a.i) : 0;
  unsigned slot_b = has1 && b.edge ? load<unsigned>(p, 30, b.i) : 0;
  T x_a = a.valid ? load<T>(p, 32, at_a) : T(0);
  T x_b = has1 && b.valid ? load<T>(p, 32, at_b) : T(0);
  asm volatile("" ::: "memory");
  finish_seg<T>(p, s, a, x_a, slot_a);
  if (has1)
    finish_seg<T>(p, s, b, x_b, slot_b);
}
template <class T> __global__ void cavities_wide(Params p) {
  unsigned *ids = buf<unsigned>(p, 24);
  if (blockIdx.x >= ids[0])
    return;
  unsigned v = ids[blockIdx.x + 1], lane = threadIdx.x;
  Control &s = *buf<Control>(p, 19);
  if (s.done)
    return;
  unsigned *vo = buf<unsigned>(p, 0);
  if (buf<signed char>(p, 5)[v] >= 0) {
    for (unsigned i = vo[2 * v] + lane; i < vo[2 * v + 2]; i += 256)
      put_q(p, i, T(-1), T(0));
    return;
  }
  T *q = buf<T>(p, 10);
  __shared__ T temp[256];
  __shared__ unsigned counts[256];
  T a[2] = {1, 1};
  unsigned d = 0;
  for (unsigned sign = 0; sign < 2; ++sign) {
    unsigned start = vo[2 * v + sign], end = vo[2 * v + sign + 1];
    T carry = 1;
    for (unsigned base = start; base < end; base += 256) {
      unsigned i = base + lane;
      T x = i < end ? packed_eta<T>(p, i) : T(0);
      temp[lane] = i < end ? T(1) - absolute(x) : T(1);
      if (i < end && p.observables)
        d += active_of(x);
      __syncthreads();
      for (unsigned step = 1; step < 256; step *= 2) {
        T earlier = lane >= step ? temp[lane - step] : T(1);
        __syncthreads();
        temp[lane] = product(temp[lane], earlier, s);
        __syncthreads();
      }
      if (i < end)
        q[i] = product(carry, lane ? temp[lane - 1] : T(1), s);
      carry = product(carry, temp[255], s);
      __syncthreads();
    }
    a[sign] = carry;
  }
  counts[lane] = d;
  __syncthreads();
  for (unsigned step = 128; step; step >>= 1) {
    if (lane < step)
      counts[lane] += counts[lane + step];
    __syncthreads();
  }
  if (!lane && p.observables) {
    buf<unsigned>(p, 7)[v] = counts[0];
    buf<Pair<T>>(p, 13)[v] = {a[0], a[1]};
  }
  for (unsigned sign = 0; sign < 2; ++sign) {
    unsigned start = vo[2 * v + sign], end = vo[2 * v + sign + 1];
    T carry = 1, opposite = a[1 - sign];
    for (unsigned done = 0; done < end - start; done += 256) {
      bool valid = done + lane < end - start;
      unsigned i = valid ? end - 1 - done - lane : start;
      temp[lane] = valid ? T(1) - absolute(packed_eta<T>(p, i)) : T(1);
      __syncthreads();
      for (unsigned step = 1; step < 256; step *= 2) {
        T earlier = lane >= step ? temp[lane - step] : T(1);
        __syncthreads();
        temp[lane] = product(temp[lane], earlier, s);
        __syncthreads();
      }
      if (valid) {
        T suffix = product(carry, lane ? temp[lane - 1] : T(1), s),
          same = product(q[i], suffix, s),
          z = opposite + (T(1) - opposite) * same;
        if (!(z > 0) || !isfinite(z))
          atomicOr(&s.error, 1u);
        else
          put_q(p, i, clamp01((T(1) - opposite) * same / z), z);
      }
      carry = product(carry, temp[255], s);
      __syncthreads();
    }
  }
}
template <class T, unsigned Width>
__device__ T clause_update(const Params &p, Control &s, unsigned c,
                           unsigned start, unsigned end) {
  if (!buf<unsigned>(p, 6)[c])
    return 0;
  unsigned *q_at = buf<unsigned>(p, 31);
  T *q = buf<T>(p, 1), *eta = buf<T>(p, 8), *scratch = buf<T>(p, 12);
  unsigned free = 0;
  T a = 1, local[Width ? Width : 1], local_q[Width ? Width : 1];
  for (unsigned e = start; e < end; ++e) {
    T value = q[q_at[e]];
    if (Width)
      local_q[e - start] = value;
    if (free_q(value)) {
      if (Width)
        local[e - start] = a;
      else
        scratch[e] = a;
      a *= value;
      ++free;
    }
  }
  if (!free) {
    atomicOr(&s.error, 2u);
    return 0;
  }
  T suffix = 1, r = 0;
  for (unsigned e = end; e > start;) {
    --e;
    T value = Width ? local_q[e - start] : q[q_at[e]];
    if (!free_q(value))
      continue;
    T raw = (Width ? local[e - start] : scratch[e]) * suffix;
    suffix *= value;
    if (raw < T(1e-16))
      raw = 0;
    r = maximum(r, absolute(raw - eta[e]));
    eta[e] = raw + (eta[e] - raw) * T(p.damping);
  }
  return r;
}
// Residuals are nonnegative, so their bit patterns order like their values
// and an atomic maximum over blocks equals the baseline reduction.
__device__ void publish_max(float r, void *slot) {
  atomicMax(static_cast<unsigned *>(slot), __float_as_uint(r));
}
__device__ void publish_max(double r, void *slot) {
  atomicMax(static_cast<unsigned long long *>(slot),
            static_cast<unsigned long long>(__double_as_longlong(r)));
}
// A bucket header with the top bit set lists a contiguous clause range: the
// first clause and its first edge follow the header.
template <class T, unsigned Width> __global__ void clauses(Params p) {
  unsigned *ids = buf<unsigned>(p, 25) + p.base, i = tid();
  Control &s = *buf<Control>(p, 19);
  T r = 0;
  if (i < (ids[0] & 0x7fffffffu) && !stopped(s)) {
    if (ids[0] >> 31) {
      unsigned start = ids[2] + i * Width;
      r = clause_update<T, Width>(p, s, ids[1] + i, start, start + Width);
    } else {
      unsigned c = ids[1 + i], *offset = buf<unsigned>(p, 2);
      r = clause_update<T, Width>(p, s, c, offset[c], offset[c + 1]);
    }
  }
  for (unsigned k = 16; k; k >>= 1)
    r = maximum(r, __shfl_xor_sync(0xffffffffu, r, k));
  __shared__ T warps[8];
  if (!(threadIdx.x & 31))
    warps[threadIdx.x / 32] = r;
  __syncthreads();
  if (!threadIdx.x) {
    for (unsigned k = 1; k < 8; ++k)
      r = maximum(r, warps[k]);
    if (r > 0)
      publish_max(r, p.b[20]);
  }
}
template <class T> __global__ void residual_final(Params p) {
  Control &s = *buf<Control>(p, 19);
  if (tid() || stopped(s))
    return;
  T *slot = buf<T>(p, 20), r = *slot;
  *slot = 0;
  if (s.error) {
    s.done = 1;
    return;
  }
  s.residual = double(r);
  ++s.sweeps;
  s.done = s.residual <= p.epsilon;
}
template <class T> __global__ void variables(Params p) {
  unsigned v = tid();
  if (v >= p.n)
    return;
  T *terms = buf<T>(p, 17);
  unsigned *flags = buf<unsigned>(p, 16);
  if (buf<signed char>(p, 5)[v] >= 0) {
    terms[v] = 0;
    flags[v] = 1;
    return;
  }
  Control &s = *buf<Control>(p, 19);
  Pair<T> a = buf<Pair<T>>(p, 13)[v];
  T z = a.x + (T(1) - a.x) * a.y;
  if (!(z > 0) || !isfinite(z)) {
    atomicOr(&s.error, 1u);
    return;
  }
  T st = (T(1) - a.x) * a.y / z, sf = (T(1) - a.y) * a.x / z;
  buf<Quad<T>>(p, 14)[v] = {st, sf, clamp01(a.x * a.y / z),
                            T(1) - (st < sf ? st : sf)};
  unsigned degree = buf<unsigned>(p, 7)[v];
  terms[v] = (T(degree) - T(1)) * logarithm(z);
  flags[v] = degree < 2 || z == 1;
}
template <class T> __global__ void clause_observables(Params p) {
  unsigned c = tid();
  if (c >= p.m)
    return;
  T *terms = buf<T>(p, 18);
  if (!buf<unsigned>(p, 6)[c]) {
    terms[c] = 0;
    return;
  }
  unsigned *offset = buf<unsigned>(p, 2), *q_at = buf<unsigned>(p, 31);
  Pair<T> *qz = buf<Pair<T>>(p, 1);
  T a = 1, sum = 0, correction = 0;
  for (unsigned e = offset[c]; e < offset[c + 1]; ++e) {
    Pair<T> value = qz[q_at[e]];
    if (free_q(value.x)) {
      a *= value.x;
      T y = logarithm(value.y) - correction, t = sum + y;
      correction = (t - sum) - y;
      sum = t;
    }
  }
  if (a >= 1) {
    atomicOr(&buf<Control>(p, 19)->error, 1u);
    return;
  }
  terms[c] = sum + log_one_plus(-a);
}
template <class T> __global__ void observable_blocks(Params p) {
  unsigned i = tid(), lane = threadIdx.x;
  __shared__ T vt[256], ct[256];
  __shared__ unsigned flags[256];
  vt[lane] = i < p.n ? buf<T>(p, 17)[i] : T(0);
  ct[lane] = i < p.m ? buf<T>(p, 18)[i] : T(0);
  flags[lane] = i < p.n ? buf<unsigned>(p, 16)[i] : 1;
  __syncthreads();
  for (unsigned k = 128; k; k >>= 1) {
    if (lane < k) {
      vt[lane] += vt[lane + k];
      ct[lane] += ct[lane + k];
      flags[lane] &= flags[lane + k];
    }
    __syncthreads();
  }
  if (!lane)
    buf<Quad<T>>(p, 23)[blockIdx.x] = {vt[0], ct[0], T(flags[0]), 0};
}
template <class T> __global__ void observable_final(Params p) {
  if (tid())
    return;
  T vs = 0, cs = 0, vc = 0, cc = 0;
  unsigned flag = 1;
  for (unsigned i = 0; i < (maximum(p.n, p.m) + 255) / 256; ++i) {
    Quad<T> a = buf<Quad<T>>(p, 23)[i];
    T y = a.x - vc, t = vs + y;
    vc = (t - vs) - y;
    vs = t;
    y = a.y - cc;
    t = cs + y;
    cc = (t - cs) - y;
    cs = t;
    flag &= a.z != 0;
  }
  Control &s = *buf<Control>(p, 19);
  s.paramagnetic = flag;
  s.complexity = flag ? 0 : double(cs - vs);
  if (!isfinite(s.complexity))
    atomicOr(&s.error, 1u);
}
__device__ unsigned domain(unsigned op) { return op < 2 ? 0 : op - 1; }
template <class T> __global__ void apply_patches(Params p) {
  unsigned i = tid();
  if (i >= p.count)
    return;
  Patch *ops = buf<Patch>(p, 27);
  Patch first = ops[i];
  if (i && ops[i - 1].index == first.index &&
      domain(ops[i - 1].op) == domain(first.op))
    return;
  for (unsigned j = i; j < p.count && ops[j].index == first.index &&
                       domain(ops[j].op) == domain(first.op);
       ++j) {
    Patch op = ops[j];
    switch (op.op) {
    case 0:
      buf<T>(p, 9)[op.index] = buf<T>(p, 8)[op.index];
      buf<T>(p, 8)[op.index] = 0;
      break;
    case 1:
      buf<T>(p, 8)[op.index] = buf<T>(p, 9)[op.index];
      break;
    case 2:
      buf<signed char>(p, 5)[op.index] = static_cast<signed char>(op.value);
      break;
    case 3:
      buf<unsigned>(p, 6)[op.index] = unsigned(op.value);
      break;
    }
  }
}

struct Context {
  void *buffers[SLOTS]{};
  size_t lengths[SLOTS]{};
  bool fp32;
  int device = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, end = nullptr;
  char name[256]{};
};
static bool checked(cudaError_t code, char *error, size_t size) {
  if (code == cudaSuccess)
    return true;
  snprintf(error, size, "%s: %s", cudaGetErrorName(code),
           cudaGetErrorString(code));
  return false;
}
extern "C" void bsp_cuda_drop(void *ptr) {
  Context *c = static_cast<Context *>(ptr);
  if (!c)
    return;
  cudaSetDevice(c->device);
  if (c->stream)
    cudaStreamSynchronize(c->stream);
  for (auto b : c->buffers)
    if (b)
      cudaFree(b);
  if (c->start)
    cudaEventDestroy(c->start);
  if (c->end)
    cudaEventDestroy(c->end);
  if (c->stream)
    cudaStreamDestroy(c->stream);
  delete c;
}
extern "C" void *bsp_cuda_new(bool fp32, char *error, size_t size) {
  Context *c = new (std::nothrow) Context;
  if (!c) {
    snprintf(error, size, "CUDA owner allocation failed");
    return nullptr;
  }
  c->fp32 = fp32;
  cudaDeviceProp props;
  if (!checked(cudaSetDevice(0), error, size) ||
      !checked(cudaGetDeviceProperties(&props, 0), error, size) ||
      !checked(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking),
               error, size) ||
      !checked(cudaEventCreate(&c->start), error, size) ||
      !checked(cudaEventCreate(&c->end), error, size)) {
    bsp_cuda_drop(c);
    return nullptr;
  }
  snprintf(c->name, sizeof(c->name), "%s (sm_%d%d)", props.name, props.major,
           props.minor);
  return c;
}
extern "C" const char *bsp_cuda_name(void *ptr) {
  return static_cast<Context *>(ptr)->name;
}
extern "C" bool bsp_cuda_budget(void *ptr, size_t *free_bytes, char *error,
                                size_t size) {
  Context *c = static_cast<Context *>(ptr);
  size_t total;
  return checked(cudaSetDevice(c->device), error, size) &&
         checked(cudaMemGetInfo(free_bytes, &total), error, size);
}
extern "C" bool bsp_cuda_buffer(void *ptr, unsigned slot, size_t length,
                                char *error, size_t size) {
  Context *c = static_cast<Context *>(ptr);
  if (slot >= SLOTS) {
    snprintf(error, size, "invalid CUDA buffer slot");
    return false;
  }
  if (!checked(cudaSetDevice(c->device), error, size) ||
      !checked(cudaStreamSynchronize(c->stream), error, size))
    return false;
  if (c->buffers[slot] && !checked(cudaFree(c->buffers[slot]), error, size))
    return false;
  c->buffers[slot] = nullptr;
  c->lengths[slot] = length;
  return checked(cudaMalloc(&c->buffers[slot], length), error, size) &&
         checked(cudaMemsetAsync(c->buffers[slot], 0, length, c->stream), error,
                 size) &&
         checked(cudaStreamSynchronize(c->stream), error, size);
}
extern "C" bool bsp_cuda_write(void *ptr, unsigned slot, const void *data,
                               size_t length, char *error, size_t size) {
  Context *c = static_cast<Context *>(ptr);
  if (slot >= SLOTS || length > c->lengths[slot]) {
    snprintf(error, size, "CUDA upload exceeds buffer");
    return false;
  }
  return checked(cudaSetDevice(c->device), error, size) &&
         checked(cudaMemcpyAsync(c->buffers[slot], data, length,
                                 cudaMemcpyHostToDevice, c->stream),
                 error, size) &&
         checked(cudaStreamSynchronize(c->stream), error, size);
}
extern "C" bool bsp_cuda_read(void *ptr, unsigned slot, void *data,
                              size_t length, char *error, size_t size) {
  Context *c = static_cast<Context *>(ptr);
  if (slot >= SLOTS || length > c->lengths[slot]) {
    snprintf(error, size, "CUDA readback exceeds buffer");
    return false;
  }
  return checked(cudaSetDevice(c->device), error, size) &&
         checked(cudaMemcpyAsync(data, c->buffers[slot], length,
                                 cudaMemcpyDeviceToHost, c->stream),
                 error, size) &&
         checked(cudaStreamSynchronize(c->stream), error, size);
}
extern "C" bool bsp_cuda_begin(void *ptr, char *error, size_t size) {
  Context *c = static_cast<Context *>(ptr);
  return checked(cudaSetDevice(c->device), error, size) &&
         checked(cudaEventRecord(c->start, c->stream), error, size);
}
extern "C" bool bsp_cuda_checkpoint(void *ptr, size_t bytes, char *error,
                                    size_t size) {
  Context *c = static_cast<Context *>(ptr);
  return checked(cudaMemcpyAsync(c->buffers[22], c->buffers[8], bytes,
                                 cudaMemcpyDeviceToDevice, c->stream),
                 error, size);
}
extern "C" bool bsp_cuda_restore_checkpoint(void *ptr, size_t bytes,
                                            char *error, size_t size) {
  Context *c = static_cast<Context *>(ptr);
  return checked(cudaSetDevice(c->device), error, size) &&
         checked(cudaMemcpyAsync(c->buffers[8], c->buffers[22], bytes,
                                 cudaMemcpyDeviceToDevice, c->stream),
                 error, size) &&
         checked(cudaStreamSynchronize(c->stream), error, size);
}
template <class T>
static void launch(Context *c, unsigned kind, Params p, unsigned count) {
  unsigned blocks =
      unsigned((static_cast<unsigned long long>(count) + 255) / 256);
  switch (kind) {
  case 0:
    cavities<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 1: {
    unsigned records = p.count >> 5;
    unsigned workers = (records + 1u) >> 1;
    unsigned threads = workers << 5;
    unsigned nblocks =
        unsigned((static_cast<unsigned long long>(threads) + 255) / 256);
    cavities_simd<T><<<nblocks, 256, 0, c->stream>>>(p);
    break;
  }
  case 2:
    cavities_wide<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 3:
    clauses<T, 3><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 4:
    clauses<T, 4><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 5:
    clauses<T, 6><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 6:
    clauses<T, 12><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 7:
    clauses<T, 0><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 9:
    residual_final<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 10:
    variables<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 11:
    clause_observables<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 12:
    observable_blocks<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 13:
    observable_final<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 14:
    apply_patches<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  case 15:
    pack_eta<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
  }
}
extern "C" bool bsp_cuda_dispatch(void *ptr, unsigned kind, unsigned count,
                                  unsigned n, unsigned m, unsigned base,
                                  unsigned observables, double epsilon,
                                  double damping, char *error, size_t size) {
  Context *c = static_cast<Context *>(ptr);
  if (kind > 15) {
    snprintf(error, size, "invalid CUDA kernel");
    return false;
  }
  if (!count)
    return true;
  Params p{};
  memcpy(p.b, c->buffers, sizeof(p.b));
  p.n = n;
  p.m = m;
  p.base = base;
  p.count = count;
  p.observables = observables;
  p.epsilon = epsilon;
  p.damping = damping;
  if (c->fp32)
    launch<float>(c, kind, p, count);
  else
    launch<double>(c, kind, p, count);
  return checked(cudaGetLastError(), error, size);
}
extern "C" bool bsp_cuda_end(void *ptr, double *seconds, char *error,
                             size_t size) {
  Context *c = static_cast<Context *>(ptr);
  float ms = 0;
  if (!checked(cudaEventRecord(c->end, c->stream), error, size) ||
      !checked(cudaStreamSynchronize(c->stream), error, size) ||
      !checked(cudaEventElapsedTime(&ms, c->start, c->end), error, size))
    return false;
  *seconds = double(ms) * 0.001;
  return true;
}
