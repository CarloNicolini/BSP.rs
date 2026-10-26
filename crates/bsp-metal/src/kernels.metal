#include <metal_stdlib>
using namespace metal;
struct Config { uint n, m; float epsilon, damping; uint observables; };
struct Control { atomic_uint error; uint done, sweeps, paramagnetic; float residual, complexity; uint unused[2]; };
inline bool stopped(device Control &s) { return s.done || atomic_load_explicit(&s.error,memory_order_relaxed); }
inline float log_one_plus(float x) { float y=1+x; return y==1 ? x : log(y)*(x/(y-1)); }
inline float checked_product(float a, float b, device Control &s) {
    float p = a*b;
    if (a>0 && b>0 && p==0) atomic_fetch_or_explicit(&s.error,4u,memory_order_relaxed);
    return p;
}
kernel void cavities(const device uint *vo [[buffer(0)]], const device uint *ve [[buffer(1)]],
    const device uint *ec [[buffer(4)]], const device char *assignment [[buffer(5)]],
    const device uint *active [[buffer(6)]], device uint *degree [[buffer(7)]],
    const device float *eta [[buffer(8)]], device float *q [[buffer(10)]],
    device float *norms [[buffer(11)]], device float2 *products [[buffer(13)]],
    device Control &state [[buffer(19)]], constant Config &cfg [[buffer(20)]], const device uint *ids [[buffer(21)]], uint id [[thread_position_in_grid]]) {
    if (id>=ids[0]) return;uint v=ids[id+1];
    if (v>=cfg.n || assignment[v]>=0 || stopped(state)) return;
    uint original_degree=vo[2*v+2]-vo[2*v];
    if (original_degree<=32 || original_degree>=128) return;
    float p[2]={1,1}; uint d=0;
    for (uint sign=0; sign<2; ++sign) {
        uint g=2*v+sign;
        for (uint i=vo[g]; i<vo[g+1]; ++i) {
            uint e=ve[i]; q[e]=p[sign];
            p[sign]=checked_product(p[sign],1-eta[e],state); if (cfg.observables) d+=active[ec[e]];
        }
    }
    if (cfg.observables) {products[v]=float2(p[0],p[1]); degree[v]=d;}
    for (uint sign=0; sign<2; ++sign) {
        uint g=2*v+sign; float suffix=1, opposite=p[1-sign];
        for (uint i=vo[g+1]; i>vo[g];) {
            --i;uint e=ve[i], qi=e; float same=checked_product(q[qi],suffix,state);
            suffix=checked_product(suffix,1-eta[e],state);
            float z=opposite+(1-opposite)*same;
            if (!(z>0) || !isfinite(z)) { atomic_fetch_or_explicit(&state.error,1u,memory_order_relaxed); return; }
            q[qi]=clamp((1-opposite)*same/z,0.0f,1.0f); if (cfg.observables) norms[qi]=z;
        }
    }
}

kernel void cavities_simd(const device uint *vo [[buffer(0)]], const device uint *ve [[buffer(1)]],
    const device uint *ec [[buffer(4)]], const device char *assignment [[buffer(5)]],
    const device uint *active [[buffer(6)]], device uint *degree [[buffer(7)]],
    const device float *eta [[buffer(8)]], device float *q [[buffer(10)]],
    device float *norms [[buffer(11)]], device float2 *products [[buffer(13)]],
    device Control &state [[buffer(19)]], constant Config &cfg [[buffer(20)]],
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint v=i/32;
    if (v>=cfg.n || assignment[v]>=0 || state.done) return;
    uint start=vo[2*v], split=vo[2*v+1], end=vo[2*v+2];
    if (end-start>32) return;
    uint p=start+lane; bool valid=p<end;uint e=valid ? ve[p] : 0;
    float factor=valid ? 1-eta[e] : 1;
    float fp=valid && p<split ? factor : 1, fm=valid && p>=split ? factor : 1;
    float plus=simd_product(fp), minus=simd_product(fm);
    if ((plus==0 && simd_all(fp>0)) || (minus==0 && simd_all(fm>0))) atomic_fetch_or_explicit(&state.error,4u,memory_order_relaxed);
    float pp=simd_prefix_exclusive_product(fp), pm=simd_prefix_exclusive_product(fm);
    float sp=simd_shuffle(simd_prefix_exclusive_product(simd_shuffle(fp,31-lane)),31-lane);
    float sm=simd_shuffle(simd_prefix_exclusive_product(simd_shuffle(fm,31-lane)),31-lane);
    uint d=cfg.observables ? simd_sum(valid ? active[ec[e]] : 0u) : 0;
    if (!lane && cfg.observables) {degree[v]=d;products[v]=float2(plus,minus);}
    if (valid) {
        float same=p<split ? pp*sp : pm*sm, opposite=p<split ? minus : plus;
        float z=opposite+(1-opposite)*same;
        if (!(z>0) || !isfinite(z)) atomic_fetch_or_explicit(&state.error,1u,memory_order_relaxed);
        else {uint qi=e;q[qi]=clamp((1-opposite)*same/z,0.0f,1.0f);if (cfg.observables) norms[qi]=z;}
    }
}

// High-degree variables use one cooperative group. Scans retain exclusion
// products for zero factors; reassociation belongs to the opt-in f32 contract.
kernel void cavities_wide(const device uint *vo [[buffer(0)]], const device uint *ve [[buffer(1)]],
    const device uint *ec [[buffer(4)]], const device char *assignment [[buffer(5)]],
    const device uint *active [[buffer(6)]], device uint *degree [[buffer(7)]],
    const device float *eta [[buffer(8)]], device float *q [[buffer(10)]],
    device float *norms [[buffer(11)]], device float2 *products [[buffer(13)]],
    device Control &state [[buffer(19)]], constant Config &cfg [[buffer(20)]], const device uint *ids [[buffer(24)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    if (group>=ids[0]) return;
    uint v=ids[group+1];
    // Only uniform conditions may bypass a group containing barriers.
    if (assignment[v]>=0 || state.done) return;
    threadgroup float temp[256]; threadgroup uint counts[256];
    float p[2]={1,1}; uint d=0;
    for (uint sign=0; sign<2; ++sign) {
        uint start=vo[2*v+sign], end=vo[2*v+sign+1]; float carry=1;
        for (uint base=start; base<end; base+=256) {
            uint i=base+lane; float factor=i<end ? 1-eta[ve[i]] : 1;
            if (i<end && cfg.observables) d+=active[ec[ve[i]]];
            temp[lane]=factor; threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint step=1; step<256; step<<=1) {
                float earlier=lane>=step ? temp[lane-step] : 1;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                temp[lane]=checked_product(temp[lane],earlier,state);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (i<end) q[ve[i]]=checked_product(carry,lane ? temp[lane-1] : 1,state);
            carry=checked_product(carry,temp[255],state);
            threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
        }
        p[sign]=carry;
    }
    counts[lane]=d; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=128; step; step>>=1) {if(lane<step) counts[lane]+=counts[lane+step];threadgroup_barrier(mem_flags::mem_threadgroup);}
    if (!lane && cfg.observables) {degree[v]=counts[0];products[v]=float2(p[0],p[1]);}
    for (uint sign=0; sign<2; ++sign) {
        uint start=vo[2*v+sign], end=vo[2*v+sign+1];float carry=1, opposite=p[1-sign];
        for (uint done=0; done<end-start; done+=256) {
            bool valid=done+lane<end-start;uint i=valid ? end-1-done-lane : start;
            temp[lane]=valid ? 1-eta[ve[i]] : 1;threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint step=1; step<256; step<<=1) {
                float earlier=lane>=step ? temp[lane-step] : 1;threadgroup_barrier(mem_flags::mem_threadgroup);
                temp[lane]=checked_product(temp[lane],earlier,state);threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (valid) {
                uint e=ve[i], qi=e;float suffix=checked_product(carry,lane ? temp[lane-1] : 1,state);
                float same=checked_product(q[qi],suffix,state), z=opposite+(1-opposite)*same;
                if (!(z>0) || !isfinite(z)) atomic_fetch_or_explicit(&state.error,1u,memory_order_relaxed);
                else {q[qi]=clamp((1-opposite)*same/z,0.0f,1.0f);if (cfg.observables) norms[qi]=z;}
            }
            carry=checked_product(carry,temp[255],state);threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
        }
    }
}

// Short clauses retain prefixes in registers; arbitrary widths use E-sized scratch.
template<uint Width> inline void clause_update(uint c, const device uint *offsets,
    const device uint *literals, const device char *assignment, const device uint *active,
    device float *eta, const device float *q, device float *scratch, device float *residual,
    device Control &state, constant Config &cfg) {
    if (stopped(state)) return;
    if (!active[c]) { residual[c]=0; return; }
    uint start=offsets[c], end=offsets[c+1], free_count=0;
    float product=1, local[Width ? Width : 1], local_q[Width ? Width : 1];
    for (uint e=start; e<end; ++e) if (assignment[literals[e]>>1]<0) {
        if (Width) local[e-start]=product; else scratch[e]=product;
        float value=q[e];
        if (Width) local_q[e-start]=value;
        product*=value; ++free_count;
    }
    if (!free_count) { atomic_fetch_or_explicit(&state.error,2u,memory_order_relaxed); return; }
    float suffix=1, r=0;
    for (uint e=end; e>start;) {
        --e; if (assignment[literals[e]>>1]>=0) continue;
        float raw=(Width ? local[e-start] : scratch[e])*suffix;
        suffix*=Width ? local_q[e-start] : q[e];
        if (raw<1e-16f) raw=0;
        r=max(r,abs(raw-eta[e])); eta[e]=raw+(eta[e]-raw)*cfg.damping;
    }
    residual[c]=r;
}
#define CLAUSE_KERNEL(Name,Width) \
kernel void Name(const device uint *offsets [[buffer(2)]], const device uint *literals [[buffer(3)]], \
    const device char *assignment [[buffer(5)]], const device uint *active [[buffer(6)]], \
    device float *eta [[buffer(8)]], const device float *q [[buffer(10)]], device float *scratch [[buffer(12)]], \
    device float *residual [[buffer(15)]], device Control &state [[buffer(19)]], constant Config &cfg [[buffer(20)]], \
    const device uint *ids [[buffer(25)]], constant uint &base [[buffer(26)]], uint i [[thread_position_in_grid]]) { \
    uint end=ids[base]; if (i>=end) return; \
    clause_update<Width>(ids[base+1+i],offsets,literals,assignment,active,eta,q,scratch,residual,state,cfg); }
CLAUSE_KERNEL(clauses3,3)
CLAUSE_KERNEL(clauses4,4)
CLAUSE_KERNEL(clauses6,6)
CLAUSE_KERNEL(clauses12,12)
CLAUSE_KERNEL(clauses_general,0)

kernel void residual_blocks(const device float *r [[buffer(15)]], device Control &s [[buffer(19)]],
    constant Config &cfg [[buffer(20)]], device float4 *blocks [[buffer(23)]],
    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    if (stopped(s)) return;
    threadgroup float temp[256]; temp[lane]=i<cfg.m ? r[i] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128; stride; stride>>=1) {
        if (lane<stride) temp[lane]=max(temp[lane],temp[lane+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!lane) blocks[group].x=temp[0];
}
kernel void residual_final(device Control &s [[buffer(19)]], constant Config &cfg [[buffer(20)]],
    const device float4 *blocks [[buffer(23)]], uint lane [[thread_index_in_threadgroup]]) {
    if (stopped(s)) return;
    float r=0; for (uint i=lane; i<(cfg.m+255)/256; i+=256) r=max(r,blocks[i].x);
    threadgroup float temp[256]; temp[lane]=r; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128; stride; stride>>=1) { if(lane<stride) temp[lane]=max(temp[lane],temp[lane+stride]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (!lane) { s.residual=temp[0]; ++s.sweeps; s.done=temp[0]<=cfg.epsilon; }
}
kernel void variables(const device char *assignment [[buffer(5)]], const device uint *degree [[buffer(7)]],
    const device float2 *products [[buffer(13)]], device float4 *surveys [[buffer(14)]],
    device uint *paramagnetic [[buffer(16)]],
    device float *terms [[buffer(17)]], device Control &s [[buffer(19)]], constant Config &cfg [[buffer(20)]], uint v [[thread_position_in_grid]]) {
    if (v>=cfg.n) return;
    if (assignment[v]>=0) { terms[v]=0;paramagnetic[v]=1; return; }
    float p=products[v].x, m=products[v].y, z=p+(1-p)*m;
    if (!(z>0) || !isfinite(z)) { atomic_fetch_or_explicit(&s.error,1u,memory_order_relaxed); return; }
    float st=(1-p)*m/z, sf=(1-m)*p/z;
    surveys[v]=float4(st,sf,clamp(p*m/z,0.0f,1.0f),1-min(st,sf));
    terms[v]=(float(degree[v])-1)*log(z);
    paramagnetic[v]=degree[v]<2 || z==1;
}
kernel void clause_observables(const device uint *offsets [[buffer(2)]], const device uint *literals [[buffer(3)]],
    const device char *assignment [[buffer(5)]], const device uint *active [[buffer(6)]],
    const device float *q [[buffer(10)]], const device float *norms [[buffer(11)]],
    device float *terms [[buffer(18)]], device Control &s [[buffer(19)]], constant Config &cfg [[buffer(20)]], uint c [[thread_position_in_grid]]) {
    if (c>=cfg.m) return;
    if (!active[c]) { terms[c]=0; return; }
    float product=1, sum=0, correction=0;
    for (uint e=offsets[c]; e<offsets[c+1]; ++e) if (assignment[literals[e]>>1]<0) {
        uint qi=e;
        product*=q[qi]; float y=log(norms[qi])-correction, t=sum+y; correction=(t-sum)-y; sum=t;
    }
    if (product>=1) { atomic_fetch_or_explicit(&s.error,1u,memory_order_relaxed); return; }
    terms[c]=sum+log_one_plus(-product);
}
kernel void observable_blocks(const device float *vt [[buffer(17)]], const device float *ct [[buffer(18)]],
    const device uint *paramagnetic [[buffer(16)]],
    constant Config &cfg [[buffer(20)]], device float4 *blocks [[buffer(23)]], uint i [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup float2 temp[256];threadgroup uint flags[256];
    temp[lane]=float2(i<cfg.n ? vt[i] : 0,i<cfg.m ? ct[i] : 0);flags[lane]=i<cfg.n ? paramagnetic[i] : 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride=128; stride; stride>>=1) { if(lane<stride) {temp[lane]+=temp[lane+stride];flags[lane]&=flags[lane+stride];} threadgroup_barrier(mem_flags::mem_threadgroup); }
    if (!lane) blocks[group]=float4(temp[0],float(flags[0]),0);
}
kernel void observable_final(device Control &s [[buffer(19)]], constant Config &cfg [[buffer(20)]],
    const device float4 *blocks [[buffer(23)]], uint i [[thread_position_in_grid]]) {
    if (i) return;
    float2 sum=0, correction=0;uint paramagnetic=1;
    for (uint j=0; j<(max(cfg.n,cfg.m)+255)/256; ++j) {
        float2 y=blocks[j].xy-correction, t=sum+y; correction=(t-sum)-y; sum=t;
        paramagnetic&=blocks[j].z!=0;
    }
    s.paramagnetic=paramagnetic;s.complexity=paramagnetic ? 0 : sum.y-sum.x;
    if (!isfinite(s.complexity)) atomic_fetch_or_explicit(&s.error,1u,memory_order_relaxed);
}
