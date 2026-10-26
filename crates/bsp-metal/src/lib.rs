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
pub struct Topology<'a> {
    pub n: usize,
    pub offsets: &'a [u32],
    pub literals: &'a [u32],
    pub edge_clause: &'a [u32],
    pub var_offsets: &'a [u32],
    pub var_edges: &'a [u32],
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
    let source = CString::new(include_str!("kernels.metal")).unwrap();
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
        let lengths = [
            t.var_offsets.len() * 4,
            edges * 4,
            t.offsets.len() * 4,
            edges * 4,
            edges * 4,
            t.n,
            m * 4,
            t.n * 4,
            edges * 4,
            edges * 4,
            edges * 4,
            edges * 4,
            edges * 4,
            t.n * 8,
            t.n * 16,
            m * 4,
            t.n * 4,
            t.n * 4,
            m * 4,
            32,
            20,
            medium.len() * 4,
            edges * 4,
            t.n.max(m).div_ceil(256).max(1) * 16,
            wide.len() * 4,
            ids.len() * 4,
        ];
        let source = CString::new(include_str!("kernels.metal")).unwrap();
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
            n: t.n,
            m,
            edges,
            wide_variables: wide.len() - 1,
            medium_variables: medium.len() - 1,
            buckets,
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
        s.write(0, t.var_offsets);
        s.write(1, t.var_edges);
        s.write(2, t.offsets);
        s.write(3, t.literals);
        s.write(4, t.edge_clause);
        s.write(5, &vec![-1i8; t.n]);
        s.write(6, &vec![1u32; m]);
        s.write(24, &wide);
        s.write(25, &ids);
        s.write(21, &medium);
        let values: Vec<f32> = eta.iter().map(|&v| v as f32).collect();
        s.write(8, &values);
        s.write(9, &values);
        Ok(s)
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
            }],
        );
        self.gpu_seconds = 0.0;
        self.phases.clear();
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
            self.dispatch("cavities", self.medium_variables, 0)?;
            self.dispatch("cavities_simd", self.n * 32, 0)?;
            self.dispatch("cavities_wide", self.wide_variables * 256, 0)?;
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
        self.dispatch("cavities", self.medium_variables, 0)?;
        self.dispatch("cavities_simd", self.n * 32, 0)?;
        self.dispatch("cavities_wide", self.wide_variables * 256, 0)?;
        self.dispatch("variables", self.n, 0)?;
        self.dispatch("clause_observables", self.m, 0)?;
        self.dispatch("observable_blocks", self.n.max(self.m), 0)?;
        self.dispatch("observable_final", 1, 0)?;
        self.finish()?;
        Ok(self.report(started))
    }
}
