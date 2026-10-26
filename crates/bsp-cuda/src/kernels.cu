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
struct Params {
  void *b[28];
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
template <class T> __device__ T exclusive_scan(T a) {
  T b = inclusive_scan(a);
  T c = __shfl_up_sync(0xffffffffu, b, 1);
  return (threadIdx.x & 31) ? c : T(1);
}

template <class T> __global__ void cavities(Params p) {
  unsigned *ids = buf<unsigned>(p, 21);
  unsigned id = tid();
  if (id >= ids[0])
    return;
  unsigned v = ids[id + 1];
  Control &s = *buf<Control>(p, 19);
  signed char *assignment = buf<signed char>(p, 5);
  if (assignment[v] >= 0 || stopped(s))
    return;
  unsigned *vo = buf<unsigned>(p, 0), *ve = buf<unsigned>(p, 1),
           *ec = buf<unsigned>(p, 4), *active = buf<unsigned>(p, 6),
           *degree = buf<unsigned>(p, 7);
  T *eta = buf<T>(p, 8), *q = buf<T>(p, 10), *norms = buf<T>(p, 11);
  Pair<T> *products = buf<Pair<T>>(p, 13);
  T a[2] = {1, 1};
  unsigned d = 0;
  for (unsigned sign = 0; sign < 2; ++sign)
    for (unsigned i = vo[2 * v + sign]; i < vo[2 * v + sign + 1]; ++i) {
      unsigned e = ve[i];
      q[e] = a[sign];
      a[sign] = product(a[sign], T(1) - eta[e], s);
      if (p.observables)
        d += active[ec[e]];
    }
  if (p.observables) {
    products[v] = {a[0], a[1]};
    degree[v] = d;
  }
  for (unsigned sign = 0; sign < 2; ++sign) {
    T suffix = 1, opposite = a[1 - sign];
    for (unsigned i = vo[2 * v + sign + 1]; i > vo[2 * v + sign];) {
      unsigned e = ve[--i];
      T same = product(q[e], suffix, s);
      suffix = product(suffix, T(1) - eta[e], s);
      T z = opposite + (T(1) - opposite) * same;
      if (!(z > 0) || !isfinite(z)) {
        atomicOr(&s.error, 1u);
        return;
      }
      q[e] = clamp01((T(1) - opposite) * same / z);
      if (p.observables)
        norms[e] = z;
    }
  }
}
template <class T> __global__ void cavities_simd(Params p) {
  unsigned v = tid() / 32, lane = threadIdx.x & 31;
  if (v >= p.n)
    return;
  Control &s = *buf<Control>(p, 19);
  if (buf<signed char>(p, 5)[v] >= 0 || s.done)
    return;
  unsigned *vo = buf<unsigned>(p, 0), *ve = buf<unsigned>(p, 1);
  unsigned start = vo[2 * v], split = vo[2 * v + 1], end = vo[2 * v + 2];
  if (end - start > 32)
    return;
  unsigned i = start + lane;
  bool valid = i < end;
  unsigned e = valid ? ve[i] : 0;
  T factor = valid ? T(1) - buf<T>(p, 8)[e] : T(1);
  T fp = valid && i < split ? factor : T(1),
    fm = valid && i >= split ? factor : T(1);
  unsigned zero_plus = __ballot_sync(0xffffffffu, fp == 0),
           zero_minus = __ballot_sync(0xffffffffu, fm == 0);
  T xp = inclusive_scan(fp), xm = inclusive_scan(fm);
  T plus = __shfl_sync(0xffffffffu, xp, 31),
    minus = __shfl_sync(0xffffffffu, xm, 31);
  if ((plus == 0 && __all_sync(0xffffffffu, fp > 0)) ||
      (minus == 0 && __all_sync(0xffffffffu, fm > 0)))
    atomicOr(&s.error, 4u);
  T pp = exclusive_scan(fp), pm = exclusive_scan(fm);
  T sp = exclusive_scan(__shfl_sync(0xffffffffu, fp, 31 - lane)),
    sm = exclusive_scan(__shfl_sync(0xffffffffu, fm, 31 - lane));
  sp = __shfl_sync(0xffffffffu, sp, 31 - lane);
  sm = __shfl_sync(0xffffffffu, sm, 31 - lane);
  if (p.observables) {
    unsigned d = valid ? buf<unsigned>(p, 6)[buf<unsigned>(p, 4)[e]] : 0;
    for (unsigned k = 16; k; k >>= 1)
      d += __shfl_down_sync(0xffffffffu, d, k);
    if (!lane) {
      buf<unsigned>(p, 7)[v] = d;
      buf<Pair<T>>(p, 13)[v] = {plus, minus};
    }
  }
  if (valid) {
    T same = i < split ? pp * sp : pm * sm, opposite = i < split ? minus : plus;
    // With exactly one zero, excluding that edge must produce a positive
    // product. The full-product check above cannot detect this underflow.
    if (factor == 0 && same == 0 &&
        __popc(i < split ? zero_plus : zero_minus) == 1)
      atomicOr(&s.error, 4u);
    T z = opposite + (T(1) - opposite) * same;
    if (!(z > 0) || !isfinite(z))
      atomicOr(&s.error, 1u);
    else {
      buf<T>(p, 10)[e] = clamp01((T(1) - opposite) * same / z);
      if (p.observables)
        buf<T>(p, 11)[e] = z;
    }
  }
}
template <class T> __global__ void cavities_wide(Params p) {
  unsigned *ids = buf<unsigned>(p, 24);
  if (blockIdx.x >= ids[0])
    return;
  unsigned v = ids[blockIdx.x + 1], lane = threadIdx.x;
  Control &s = *buf<Control>(p, 19);
  if (buf<signed char>(p, 5)[v] >= 0 || s.done)
    return;
  unsigned *vo = buf<unsigned>(p, 0), *ve = buf<unsigned>(p, 1);
  T *eta = buf<T>(p, 8), *q = buf<T>(p, 10);
  __shared__ T temp[256];
  __shared__ unsigned counts[256];
  T a[2] = {1, 1};
  unsigned d = 0;
  for (unsigned sign = 0; sign < 2; ++sign) {
    unsigned start = vo[2 * v + sign], end = vo[2 * v + sign + 1];
    T carry = 1;
    for (unsigned base = start; base < end; base += 256) {
      unsigned i = base + lane;
      temp[lane] = i < end ? T(1) - eta[ve[i]] : T(1);
      if (i < end && p.observables)
        d += buf<unsigned>(p, 6)[buf<unsigned>(p, 4)[ve[i]]];
      __syncthreads();
      for (unsigned step = 1; step < 256; step *= 2) {
        T earlier = lane >= step ? temp[lane - step] : T(1);
        __syncthreads();
        temp[lane] = product(temp[lane], earlier, s);
        __syncthreads();
      }
      if (i < end)
        q[ve[i]] = product(carry, lane ? temp[lane - 1] : T(1), s);
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
      temp[lane] = valid ? T(1) - eta[ve[i]] : T(1);
      __syncthreads();
      for (unsigned step = 1; step < 256; step *= 2) {
        T earlier = lane >= step ? temp[lane - step] : T(1);
        __syncthreads();
        temp[lane] = product(temp[lane], earlier, s);
        __syncthreads();
      }
      if (valid) {
        unsigned e = ve[i];
        T suffix = product(carry, lane ? temp[lane - 1] : T(1), s),
          same = product(q[e], suffix, s),
          z = opposite + (T(1) - opposite) * same;
        if (!(z > 0) || !isfinite(z))
          atomicOr(&s.error, 1u);
        else {
          q[e] = clamp01((T(1) - opposite) * same / z);
          if (p.observables)
            buf<T>(p, 11)[e] = z;
        }
      }
      carry = product(carry, temp[255], s);
      __syncthreads();
    }
  }
}
template <class T, unsigned Width> __global__ void clauses(Params p) {
  unsigned *ids = buf<unsigned>(p, 25), i = tid();
  if (i >= ids[p.base])
    return;
  unsigned c = ids[p.base + 1 + i];
  Control &s = *buf<Control>(p, 19);
  if (stopped(s))
    return;
  T *residual = buf<T>(p, 15);
  if (!buf<unsigned>(p, 6)[c]) {
    residual[c] = 0;
    return;
  }
  unsigned *offset = buf<unsigned>(p, 2), *lit = buf<unsigned>(p, 3);
  signed char *assignment = buf<signed char>(p, 5);
  T *q = buf<T>(p, 10), *eta = buf<T>(p, 8), *scratch = buf<T>(p, 12);
  unsigned start = offset[c], end = offset[c + 1], free = 0;
  T a = 1, local[Width ? Width : 1], local_q[Width ? Width : 1];
  for (unsigned e = start; e < end; ++e)
    if (assignment[lit[e] >> 1] < 0) {
      if (Width) {
        local[e - start] = a;
        local_q[e - start] = q[e];
      } else
        scratch[e] = a;
      a *= q[e];
      ++free;
    }
  if (!free) {
    atomicOr(&s.error, 2u);
    return;
  }
  T suffix = 1, r = 0;
  for (unsigned e = end; e > start;) {
    --e;
    if (assignment[lit[e] >> 1] >= 0)
      continue;
    T raw = (Width ? local[e - start] : scratch[e]) * suffix;
    suffix *= Width ? local_q[e - start] : q[e];
    if (raw < T(1e-16))
      raw = 0;
    r = maximum(r, absolute(raw - eta[e]));
    eta[e] = raw + (eta[e] - raw) * T(p.damping);
  }
  residual[c] = r;
}
template <class T> __global__ void residual_blocks(Params p) {
  Control &s = *buf<Control>(p, 19);
  if (stopped(s) || s.error)
    return;
  unsigned i = tid(), lane = threadIdx.x;
  __shared__ T temp[256];
  temp[lane] = i < p.m ? buf<T>(p, 15)[i] : T(0);
  __syncthreads();
  for (unsigned k = 128; k; k >>= 1) {
    if (lane < k)
      temp[lane] = maximum(temp[lane], temp[lane + k]);
    __syncthreads();
  }
  if (!lane)
    buf<Quad<T>>(p, 23)[blockIdx.x].x = temp[0];
}
template <class T> __global__ void residual_final(Params p) {
  Control &s = *buf<Control>(p, 19);
  if (stopped(s))
    return;
  unsigned lane = threadIdx.x;
  if (s.error) {
    // Every lane has finished reading done before its single writer sets it.
    // Later sweeps in this submission must leave the failed state untouched.
    __syncthreads();
    if (!lane)
      s.done = 1;
    return;
  }
  T r = 0;
  for (unsigned i = lane; i < (p.m + 255) / 256; i += 256)
    r = maximum(r, buf<Quad<T>>(p, 23)[i].x);
  __shared__ T temp[256];
  temp[lane] = r;
  __syncthreads();
  for (unsigned k = 128; k; k >>= 1) {
    if (lane < k)
      temp[lane] = maximum(temp[lane], temp[lane + k]);
    __syncthreads();
  }
  if (!lane) {
    s.residual = double(temp[0]);
    ++s.sweeps;
    s.done = s.residual <= p.epsilon;
  }
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
  unsigned *offset = buf<unsigned>(p, 2), *lit = buf<unsigned>(p, 3);
  T *q = buf<T>(p, 10), *norm = buf<T>(p, 11);
  T a = 1, sum = 0, correction = 0;
  for (unsigned e = offset[c]; e < offset[c + 1]; ++e)
    if (buf<signed char>(p, 5)[lit[e] >> 1] < 0) {
      a *= q[e];
      T y = logarithm(norm[e]) - correction, t = sum + y;
      correction = (t - sum) - y;
      sum = t;
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
  void *buffers[28]{};
  size_t lengths[28]{};
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
  if (slot >= 28) {
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
  if (slot >= 28 || length > c->lengths[slot]) {
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
  if (slot >= 28 || length > c->lengths[slot]) {
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
  case 1:
    cavities_simd<T><<<blocks, 256, 0, c->stream>>>(p);
    break;
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
  case 8:
    residual_blocks<T><<<blocks, 256, 0, c->stream>>>(p);
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
  }
}
extern "C" bool bsp_cuda_dispatch(void *ptr, unsigned kind, unsigned count,
                                  unsigned n, unsigned m, unsigned base,
                                  unsigned observables, double epsilon,
                                  double damping, char *error, size_t size) {
  Context *c = static_cast<Context *>(ptr);
  if (kind > 14) {
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
