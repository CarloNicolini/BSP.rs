//! Synchronous, bounds-checked compute boundary. No GPU work outlives a call.
use std::{
    ffi::{c_char, c_void, CStr, CString},
    ptr::NonNull,
    time::Instant,
};

extern "C" {
    fn bsp_metal_new(source: *const c_char, error: *mut c_char, size: usize) -> *mut c_void;
    fn bsp_metal_drop(ctx: *mut c_void);
    fn bsp_metal_budget(ctx: *mut c_void) -> u64;
    fn bsp_metal_name(ctx: *mut c_void) -> *const c_char;
    fn bsp_metal_buffer(ctx: *mut c_void, length: usize) -> *mut c_void;
    fn bsp_metal_begin(ctx: *mut c_void, checkpoint: bool, bytes: usize);
    fn bsp_metal_dispatch(ctx: *mut c_void, kernel: *const c_char, count: usize, base: u32);
    fn bsp_metal_end(ctx: *mut c_void, seconds: *mut f64, error: *mut c_char, size: usize) -> bool;
}
// Edges per source window. A gathered window (2 MiB of eta) stays in the
// system-level cache; scattered q needs a smaller window (256 KiB).
const ETA_WINDOW: usize = 1 << 19;
const Q_WINDOW: usize = 1 << 16;
pub struct Topology<'a> {
    pub n: usize,
    pub offsets: &'a [u32],
    pub literals: &'a [u32],
    pub edge_clause: &'a [u32],
    pub var_offsets: &'a [u32],
    pub var_edges: &'a [u32],
}
pub struct FeistelTopology<'a> {
    pub n: usize,
    pub m: usize,
    pub k: usize,
    pub sign_key: u64,
    pub round_keys: &'a [[u64; 8]],
    pub fluctuating: Option<FeistelRepairs<'a>>,
}
pub struct FeistelRepairs<'a> {
    pub stub_offsets: &'a [u32],
    pub clauses: &'a [u32],
    // Sorted by variable and then clause-major edge ID.
    pub insertions: &'a [[u32; 2]],
}
#[derive(Clone, Copy)]
#[repr(C)]
struct FeistelConfig {
    sign_key: u64,
    half_bits: u32,
    fluctuating: u32,
    repairs: u32,
    unused: u32,
}
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Control {
    error: u32,
    done: u32,
    sweeps: u32,
    paramagnetic: u32,
    residual: f32,
    complexity: f32,
    unused: [u32; 2],
}
#[derive(Clone, Copy)]
#[repr(C)]
struct Config {
    n: u32,
    m: u32,
    epsilon: f32,
    damping: f32,
    observables: u32,
    width: u32,
}
#[derive(Default)]
pub struct Report {
    pub sweeps: usize,
    pub residual: f64,
    pub complexity: f64,
    pub paramagnetic: bool,
    pub numerical_error: bool,
    pub contradiction: bool,
    pub gpu_seconds: f64,
    pub wall_seconds: f64,
    pub phases: Vec<(String, f64)>,
}
struct Buffer {
    ptr: NonNull<u8>,
    bytes: usize,
}
pub struct MetalSolver {
    context: NonNull<c_void>,
    buffers: Vec<Buffer>,
    n: usize,
    m: usize,
    edges: usize,
    feistel_width: u32,
    wide_variables: usize,
    medium_variables: usize,
    buckets: Vec<(String, usize, u32)>,
    pub device: String,
    pub allocated_bytes: usize,
    profile: bool,
    gpu_seconds: f64,
    phases: Vec<(String, f64)>,
}
// Calls exclusively borrow the owner and complete before returning. Buffers never
// alias another owner. The shared ObjC runtime/queue supports concurrent encoding.
unsafe impl Send for MetalSolver {}
unsafe impl Sync for MetalSolver {}
impl Drop for MetalSolver {
    fn drop(&mut self) {
        unsafe { bsp_metal_drop(self.context.as_ptr()) }
    }
}

pub fn safe_conversion(values: &[f64]) -> bool {
    values.iter().all(|&v| {
        v.is_finite()
            && (0.0..=1.0).contains(&v)
            && !(v > 0.0 && (v as f32) == 0.0 || v < 1.0 && (v as f32) == 1.0)
    })
}
pub fn device_budget() -> Result<u64, String> {
    let source = CString::new(include_str!("kernels_packed.metal")).unwrap();
    let mut error = [0i8; 4096];
    let p = unsafe { bsp_metal_new(source.as_ptr(), error.as_mut_ptr(), error.len()) };
    if p.is_null() {
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned());
    }
    let budget = unsafe { bsp_metal_budget(p) };
    unsafe { bsp_metal_drop(p) };
    Ok(budget)
}
impl MetalSolver {
    pub fn new(t: Topology<'_>, eta: &[f64], profile: bool) -> Result<Self, String> {
        let edges = t.literals.len();
        let m = t
            .offsets
            .len()
            .checked_sub(1)
            .ok_or("missing clause offsets")?;
        if t.n > (u32::MAX / 32) as usize
            || edges >= u32::MAX as usize
            || m > (u32::MAX - 255) as usize
            || t.offsets[0] != 0
            || t.offsets[m] as usize != edges
            || t.offsets.windows(2).any(|w| w[0] > w[1])
            || t.edge_clause.len() != edges
            || t.var_edges.len() != edges
            || t.var_offsets.len() != 2 * t.n + 1
            || t.var_offsets[0] != 0
            || t.var_offsets[2 * t.n] as usize != edges
            || t.var_offsets.windows(2).any(|w| w[0] > w[1])
            || eta.len() != edges
            || !safe_conversion(eta)
        {
            return Err("invalid topology or unsafe f32 conversion".into());
        }
        let mut seen = vec![false; edges];
        for g in 0..2 * t.n {
            for &e in &t.var_edges[t.var_offsets[g] as usize..t.var_offsets[g + 1] as usize] {
                let e = e as usize;
                if e >= edges || seen[e] || t.literals[e] as usize != g {
                    return Err("invalid variable CSR".into());
                }
                seen[e] = true;
            }
        }
        for c in 0..m {
            for e in t.offsets[c] as usize..t.offsets[c + 1] as usize {
                if t.edge_clause[e] as usize != c || (t.literals[e] >> 1) as usize >= t.n {
                    return Err("invalid clause CSR".into());
                }
            }
        }
        let mut ids = Vec::with_capacity(m + 5);
        let mut buckets = Vec::new();
        let mut wide = vec![0u32];
        let mut medium = vec![0u32];
        for v in 0..t.n {
            if t.var_offsets[2 * v + 2] - t.var_offsets[2 * v] >= 128 {
                wide.push(v as u32);
            }
        }
        for v in 0..t.n {
            if (33..128).contains(&(t.var_offsets[2 * v + 2] - t.var_offsets[2 * v])) {
                medium.push(v as u32);
            }
        }
        medium[0] = (medium.len() - 1) as u32;
        wide[0] = (wide.len() - 1) as u32;
        for (width, name) in [
            (3, "clauses3"),
            (4, "clauses4"),
            (6, "clauses6"),
            (12, "clauses12"),
            (0, "clauses_general"),
        ] {
            let base = ids.len();
            ids.push(0);
            for c in 0..m {
                let k = t.offsets[c + 1] - t.offsets[c];
                if k == width || width == 0 && ![3, 4, 6, 12].contains(&k) {
                    ids.push(c as u32);
                }
            }
            let count = ids.len() - base - 1;
            ids[base] = count as u32;
            buckets.push((name.to_string(), count, base as u32));
        }
        // Specialized widths keep prefixes in registers. Only the generic
        // clause kernel indexes the edge-sized prefix scratch.
        let general_edges = if buckets
            .iter()
            .any(|(name, count, _)| name == "clauses_general" && *count > 0)
        {
            edges
        } else {
            0
        };
        // Only medium and wide cavities keep unpacked q as per-variable scratch.
        let scratch_q = if medium.len() > 1 || wide.len() > 1 {
            edges
        } else {
            0
        };
        let mut lengths = Self::buffer_lengths(t.n, m, edges);
        lengths[0] = t.var_offsets.len() * 4;
        lengths[2] = t.offsets.len() * 4;
        lengths[4] = edges * 4;
        lengths[10] = scratch_q * 4;
        lengths[11] = edges * 4;
        lengths[12] = general_edges * 4;
        lengths[21] = medium.len() * 4;
        lengths[24] = wide.len() * 4;
        lengths[25] = ids.len() * 4;
        lengths[27..=30].fill(edges * 4);
        let mut s = Self::allocate(t.n, m, edges, lengths, profile)?;
        s.wide_variables = wide.len() - 1;
        s.medium_variables = medium.len() - 1;
        s.buckets = buckets;
        s.write(0, t.var_offsets);
        s.write(2, t.offsets);
        s.write(4, t.edge_clause);
        s.pack_layout(t.var_edges);
        s.write(24, &wide);
        s.write(25, &ids);
        s.write(21, &medium);
        s.initialize(eta);
        Ok(s)
    }
    pub fn new_feistel(t: FeistelTopology<'_>, eta: &[f64], profile: bool) -> Result<Self, String> {
        let edges = t.m.checked_mul(t.k).ok_or("invalid Feistel dimensions")?;
        if !(1..=64).contains(&t.k)
            || t.k > t.n
            || t.n > (u32::MAX / 32) as usize
            || t.m > (u32::MAX - 255) as usize
            || edges >= u32::MAX as usize
            || t.round_keys.len() != if t.fluctuating.is_some() { 1 } else { t.k }
            || eta.len() != edges
            || !safe_conversion(eta)
        {
            return Err("invalid Feistel topology or unsafe f32 conversion".into());
        }
        let mut lengths = Self::buffer_lengths(t.n, t.m, edges);
        if let Some(f) = &t.fluctuating {
            if f.stub_offsets.len() != t.n + 1
                || f.stub_offsets[0] != 0
                || f.stub_offsets[t.n] as usize != edges
                || f.stub_offsets.windows(2).any(|w| w[0] > w[1])
                || f.clauses.len() > t.m
                || f.clauses.last().is_some_and(|&c| c as usize >= t.m)
                || f.clauses.windows(2).any(|w| w[0] >= w[1])
                || f.insertions.len() != f.clauses.len() * t.k
                || f.insertions.windows(2).any(|w| w[0] >= w[1])
            {
                return Err("invalid Feistel stub or repair table".into());
            }
            // Every repaired edge has one writer. Validate only the sparse patch.
            let mut seen = vec![false; f.insertions.len()];
            for &[v, e] in f.insertions {
                let e = e as usize;
                let row = f
                    .clauses
                    .binary_search(&((e / t.k) as u32))
                    .map_err(|_| "invalid Feistel insertion")?;
                if v as usize >= t.n || std::mem::replace(&mut seen[row * t.k + e % t.k], true) {
                    return Err("invalid Feistel insertion".into());
                }
            }
            lengths[0] = std::mem::size_of_val(f.stub_offsets);
            lengths[2] = std::mem::size_of_val(f.clauses);
            lengths[4] = std::mem::size_of_val(f.insertions);
        }
        lengths[11] = std::mem::size_of_val(t.round_keys);
        lengths[21] = std::mem::size_of::<FeistelConfig>();
        let limit = if t.fluctuating.is_some() { edges } else { t.m } as u32;
        let mut s = Self::allocate(t.n, t.m, edges, lengths, profile)?;
        s.feistel_width = t.k as u32;
        let clause_kernel = match t.k {
            3 => "clauses3",
            4 => "clauses4",
            6 => "clauses6",
            12 => "clauses12",
            _ => "clauses_general",
        };
        s.buckets.push((clause_kernel.into(), t.m, edges as u32));
        s.write(11, t.round_keys);
        s.write(
            21,
            &[FeistelConfig {
                sign_key: t.sign_key,
                half_bits: (u32::BITS - limit.saturating_sub(1).leading_zeros())
                    .div_ceil(2)
                    .max(1),
                fluctuating: u32::from(t.fluctuating.is_some()),
                repairs: t.fluctuating.as_ref().map_or(0, |f| f.clauses.len() as u32),
                unused: 0,
            }],
        );
        if let Some(f) = t.fluctuating {
            s.write(0, f.stub_offsets);
            s.write(2, f.clauses);
            s.write(4, f.insertions);
        }
        s.initialize(eta);
        Ok(s)
    }
    fn buffer_lengths(n: usize, m: usize, edges: usize) -> [usize; 31] {
        let mut lengths = [0; 31];
        lengths[1] = edges * 8; // q; observables pair each q with its normalizer.
        lengths[5..=9].copy_from_slice(&[n, m * 4, n * 4, edges * 4, edges * 4]);
        lengths[13..=20].copy_from_slice(&[
            n * 8,
            n * 16,
            m * 4,
            n * 4,
            n * 4,
            m * 4,
            std::mem::size_of::<Control>(),
            std::mem::size_of::<Config>(),
        ]);
        lengths[22] = edges * 4;
        lengths[23] = n.max(m).div_ceil(256).max(1) * 16;
        lengths
    }
    fn allocate(
        n: usize,
        m: usize,
        edges: usize,
        lengths: [usize; 31],
        profile: bool,
    ) -> Result<Self, String> {
        let source = CString::new(include_str!("kernels_packed.metal")).unwrap();
        let mut error = [0i8; 4096];
        let context = NonNull::new(unsafe {
            bsp_metal_new(source.as_ptr(), error.as_mut_ptr(), error.len())
        })
        .ok_or_else(|| {
            unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })?;
        let mut s = Self {
            context,
            buffers: Vec::new(),
            n,
            m,
            edges,
            feistel_width: 0,
            wide_variables: 0,
            medium_variables: 0,
            buckets: Vec::new(),
            device: unsafe { CStr::from_ptr(bsp_metal_name(context.as_ptr())) }
                .to_string_lossy()
                .into_owned(),
            allocated_bytes: lengths.iter().map(|&n| n.max(4)).sum(),
            profile,
            gpu_seconds: 0.0,
            phases: Vec::new(),
        };
        if s.allocated_bytes as u64 > unsafe { bsp_metal_budget(context.as_ptr()) } * 3 / 4 {
            return Err("Metal state exceeds 75% of the recommended GPU working set".into());
        }
        for bytes in lengths {
            let ptr = NonNull::new(unsafe { bsp_metal_buffer(context.as_ptr(), bytes) }.cast())
                .ok_or("Metal buffer allocation failed")?;
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0, bytes.max(4));
            }
            s.buffers.push(Buffer {
                ptr,
                bytes: bytes.max(4),
            });
        }
        Ok(s)
    }
    fn initialize(&mut self, eta: &[f64]) {
        let (n, m) = (self.n, self.m);
        self.slice_mut::<i8>(5)[..n].fill(-1);
        self.slice_mut::<u32>(6)[..m].fill(1);
        for slot in [8, 9] {
            for (dst, &v) in self.slice_mut::<f32>(slot).iter_mut().zip(eta) {
                *dst = v as f32;
            }
        }
    }
    fn slice<T>(&self, slot: usize) -> &[T] {
        let b = &self.buffers[slot];
        assert_eq!(b.ptr.as_ptr() as usize % std::mem::align_of::<T>(), 0);
        unsafe {
            std::slice::from_raw_parts(b.ptr.as_ptr().cast(), b.bytes / std::mem::size_of::<T>())
        }
    }
    fn slice_mut<T>(&mut self, slot: usize) -> &mut [T] {
        let b = &self.buffers[slot];
        assert_eq!(b.ptr.as_ptr() as usize % std::mem::align_of::<T>(), 0);
        unsafe {
            std::slice::from_raw_parts_mut(
                b.ptr.as_ptr().cast(),
                b.bytes / std::mem::size_of::<T>(),
            )
        }
    }
    fn write<T: Copy>(&mut self, slot: usize, values: &[T]) {
        self.slice_mut::<T>(slot)[..values.len()].copy_from_slice(values);
    }
    /// Random accesses outside the system-level cache dominate a sweep, so
    /// each order change is confined to aligned windows. Packed slot k lies in
    /// the bucket of its source window and follows the consumer's order there.
    fn pack_layout(&mut self, var_edges: &[u32]) {
        let mut position = vec![0u32; self.edges];
        let mut eta_at = vec![0u32; self.edges];
        let mut eta_source = vec![0u32; self.edges];
        let mut next: Vec<u32> = (0..self.edges)
            .step_by(ETA_WINDOW)
            .map(|k| k as u32)
            .collect();
        for (p, &e) in var_edges.iter().enumerate() {
            position[e as usize] = p as u32;
            let k = &mut next[e as usize / ETA_WINDOW];
            eta_at[p] = *k;
            eta_source[*k as usize] = e;
            *k += 1;
        }
        self.write(11, &eta_source);
        self.write(27, &eta_at);
        let (mut q_slot, mut q_at) = (eta_source, eta_at);
        next = (0..self.edges)
            .step_by(Q_WINDOW)
            .map(|k| k as u32)
            .collect();
        for (e, &p) in position.iter().enumerate() {
            let k = &mut next[p as usize / Q_WINDOW];
            q_slot[p as usize] = *k;
            q_at[e] = *k;
            *k += 1;
        }
        self.write(28, &q_slot);
        self.write(29, &q_at);
    }
    pub fn set_assignment(&mut self, v: usize, value: i8) {
        assert!(v < self.n);
        self.slice_mut::<i8>(5)[v] = value;
    }
    pub fn set_active(&mut self, c: usize, value: bool) {
        assert!(c < self.m);
        self.slice_mut::<u32>(6)[c] = u32::from(value);
    }
    pub fn freeze(&mut self, e: usize) {
        assert!(e < self.edges);
        let v = self.slice::<f32>(8)[e];
        self.slice_mut::<f32>(9)[e] = v;
        self.slice_mut::<f32>(8)[e] = 0.0;
    }
    pub fn restore(&mut self, e: usize) {
        assert!(e < self.edges);
        let v = self.slice::<f32>(9)[e];
        self.slice_mut::<f32>(8)[e] = v;
    }
    pub fn upload(&mut self, eta: &[f64], saved: &[f64]) -> Result<(), String> {
        if eta.len() != self.edges
            || saved.len() != self.edges
            || !safe_conversion(eta)
            || !safe_conversion(saved)
        {
            return Err("unsafe f32 message conversion".into());
        }
        for (slot, values) in [(8, eta), (9, saved)] {
            for (dst, &v) in self.slice_mut::<f32>(slot).iter_mut().zip(values) {
                *dst = v as f32;
            }
        }
        Ok(())
    }
    pub fn snapshot(&self, eta: &mut [f64], saved: &mut [f64]) {
        assert_eq!(eta.len(), self.edges);
        assert_eq!(saved.len(), self.edges);
        for (slot, values) in [(8, eta), (9, saved)] {
            for (dst, &v) in values.iter_mut().zip(self.slice::<f32>(slot)) {
                *dst = v as f64;
            }
        }
    }
    pub fn surveys(&self, out: &mut [[f64; 3]], cert: &mut [f64]) {
        assert_eq!(out.len(), self.n);
        assert_eq!(cert.len(), self.n);
        let assignments = self.slice::<i8>(5);
        for (v, data) in self.slice::<[f32; 4]>(14).iter().take(self.n).enumerate() {
            if assignments[v] < 0 {
                out[v] = [data[0] as f64, data[1] as f64, data[2] as f64];
                cert[v] = data[3] as f64;
            }
        }
    }
    pub fn restore_checkpoint(&mut self) {
        for e in 0..self.edges {
            self.slice_mut::<f32>(8)[e] = self.slice::<f32>(22)[e];
        }
    }
    fn begin(&mut self, checkpoint: bool) {
        unsafe {
            bsp_metal_begin(self.context.as_ptr(), checkpoint, self.edges * 4);
        }
    }
    fn finish(&mut self) -> Result<f64, String> {
        let mut seconds = 0.0;
        let mut error = [0i8; 2048];
        if !unsafe {
            bsp_metal_end(
                self.context.as_ptr(),
                &mut seconds,
                error.as_mut_ptr(),
                error.len(),
            )
        } {
            return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        self.gpu_seconds += seconds;
        Ok(seconds)
    }
    fn dispatch(&mut self, name: &str, count: usize, base: u32) -> Result<(), String> {
        if count == 0 {
            return Ok(());
        }
        let name_c = CString::new(name).unwrap();
        unsafe {
            bsp_metal_dispatch(self.context.as_ptr(), name_c.as_ptr(), count, base);
        }
        if self.profile {
            let seconds = self.finish()?;
            self.phases.push((name.into(), seconds));
            self.begin(false);
        }
        Ok(())
    }
    fn reset(&mut self, epsilon: f64, damping: f64) {
        self.write(
            19,
            &[Control {
                paramagnetic: 1,
                ..Control::default()
            }],
        );
        self.write(
            20,
            &[Config {
                n: self.n as u32,
                m: self.m as u32,
                epsilon: epsilon as f32,
                damping: damping as f32,
                observables: 0,
                width: self.feistel_width,
            }],
        );
        self.gpu_seconds = 0.0;
        self.phases.clear();
    }
    fn cavities(&mut self) -> Result<(), String> {
        if self.feistel_width != 0 {
            return self.dispatch("cavities_feistel", self.n, 0);
        }
        self.dispatch("pack_eta", self.edges.div_ceil(4), self.edges as u32)?;
        self.dispatch("cavities", self.medium_variables, 0)?;
        self.dispatch("cavities_simd", self.n * 32, 0)?;
        self.dispatch("cavities_wide", self.wide_variables * 256, 0)
    }
    fn report(&mut self, started: Instant) -> Report {
        let s = self.slice::<Control>(19)[0];
        Report {
            sweeps: s.sweeps as usize,
            residual: s.residual as f64,
            complexity: s.complexity as f64,
            paramagnetic: s.paramagnetic != 0,
            numerical_error: s.error & 5 != 0,
            contradiction: s.error & 2 != 0,
            gpu_seconds: self.gpu_seconds,
            wall_seconds: started.elapsed().as_secs_f64(),
            phases: std::mem::take(&mut self.phases),
        }
    }
    pub fn sweeps(
        &mut self,
        count: usize,
        epsilon: f64,
        damping: f64,
        checkpoint: bool,
    ) -> Result<Report, String> {
        let started = Instant::now();
        self.reset(epsilon, damping);
        self.begin(checkpoint);
        for _ in 0..count {
            self.cavities()?;
            for i in 0..self.buckets.len() {
                let (name, count, base) = self.buckets[i].clone();
                self.dispatch(&name, count, base)?;
            }
            self.dispatch("residual_blocks", self.m, 0)?;
            self.dispatch("residual_final", 256, 0)?;
        }
        self.finish()?;
        Ok(self.report(started))
    }
    pub fn observables(&mut self) -> Result<Report, String> {
        let started = Instant::now();
        self.reset(0.0, 0.0);
        self.slice_mut::<Config>(20)[0].observables = 1;
        self.begin(false);
        self.cavities()?;
        self.dispatch("variables", self.n, 0)?;
        self.dispatch("clause_observables", self.m, 0)?;
        self.dispatch("observable_blocks", self.n.max(self.m), 0)?;
        self.dispatch("observable_final", 1, 0)?;
        self.finish()?;
        Ok(self.report(started))
    }
}
