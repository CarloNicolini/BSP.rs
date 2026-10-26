//! Bounds-checked synchronous CUDA boundary. Device memory stays private.
use std::{
    ffi::{c_char, c_void, CStr},
    ptr::NonNull,
    time::Instant,
};
extern "C" {
    fn bsp_cuda_new(fp32: bool, error: *mut c_char, size: usize) -> *mut c_void;
    fn bsp_cuda_drop(ctx: *mut c_void);
    fn bsp_cuda_name(ctx: *mut c_void) -> *const c_char;
    fn bsp_cuda_budget(ctx: *mut c_void, free: *mut usize, error: *mut c_char, size: usize)
        -> bool;
    fn bsp_cuda_buffer(
        ctx: *mut c_void,
        slot: u32,
        length: usize,
        error: *mut c_char,
        size: usize,
    ) -> bool;
    fn bsp_cuda_write(
        ctx: *mut c_void,
        slot: u32,
        data: *const c_void,
        length: usize,
        error: *mut c_char,
        size: usize,
    ) -> bool;
    fn bsp_cuda_read(
        ctx: *mut c_void,
        slot: u32,
        data: *mut c_void,
        length: usize,
        error: *mut c_char,
        size: usize,
    ) -> bool;
    fn bsp_cuda_begin(ctx: *mut c_void, error: *mut c_char, size: usize) -> bool;
    fn bsp_cuda_checkpoint(ctx: *mut c_void, bytes: usize, error: *mut c_char, size: usize)
        -> bool;
    fn bsp_cuda_restore_checkpoint(
        ctx: *mut c_void,
        bytes: usize,
        error: *mut c_char,
        size: usize,
    ) -> bool;
    fn bsp_cuda_dispatch(
        ctx: *mut c_void,
        kind: u32,
        count: u32,
        n: u32,
        m: u32,
        base: u32,
        observables: u32,
        epsilon: f64,
        damping: f64,
        error: *mut c_char,
        size: usize,
    ) -> bool;
    fn bsp_cuda_end(ctx: *mut c_void, seconds: *mut f64, error: *mut c_char, size: usize) -> bool;
}
fn checked(action: impl FnOnce(*mut c_char, usize) -> bool) -> Result<(), String> {
    let mut error = [0i8; 2048];
    if action(error.as_mut_ptr(), error.len()) {
        Ok(())
    } else {
        Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }
}
pub struct Topology<'a> {
    pub n: usize,
    pub offsets: &'a [u32],
    pub literals: &'a [u32],
    pub edge_clause: &'a [u32],
    pub var_offsets: &'a [u32],
    pub var_edges: &'a [u32],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Control {
    error: u32,
    done: u32,
    sweeps: u32,
    paramagnetic: u32,
    residual: f64,
    complexity: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Patch {
    index: u32,
    op: u32,
    value: i32,
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
pub struct CudaSolver {
    context: NonNull<c_void>,
    lengths: Vec<usize>,
    n: usize,
    m: usize,
    edges: usize,
    fp32: bool,
    medium: usize,
    wide: usize,
    buckets: Vec<(usize, usize, u32)>,
    assignments: Vec<i8>,
    patches: Vec<Patch>,
    profile: bool,
    epsilon: f64,
    damping: f64,
    observables: bool,
    gpu_seconds: f64,
    phases: Vec<(String, f64)>,
    pub device: String,
    pub allocated_bytes: usize,
}
// Each owner has a separate stream and allocations. Calls select its device,
// borrow it exclusively, and synchronize before returning or freeing resources.
unsafe impl Send for CudaSolver {}
unsafe impl Sync for CudaSolver {}
impl Drop for CudaSolver {
    fn drop(&mut self) {
        unsafe { bsp_cuda_drop(self.context.as_ptr()) }
    }
}
fn valid_messages(values: &[f64]) -> bool {
    values
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
}
pub fn safe_conversion(values: &[f64]) -> bool {
    valid_messages(values)
        && values
            .iter()
            .all(|&v| !(v > 0.0 && v as f32 == 0.0 || v < 1.0 && v as f32 == 1.0))
}
pub fn device_budget() -> Result<u64, String> {
    let mut error = [0i8; 2048];
    let ptr = NonNull::new(unsafe { bsp_cuda_new(false, error.as_mut_ptr(), error.len()) })
        .ok_or_else(|| {
            unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })?;
    let mut free = 0;
    let result = checked(|e, z| unsafe { bsp_cuda_budget(ptr.as_ptr(), &mut free, e, z) });
    unsafe { bsp_cuda_drop(ptr.as_ptr()) };
    result.map(|()| free as u64)
}
impl CudaSolver {
    pub fn new(t: Topology<'_>, eta: &[f64], fp32: bool, profile: bool) -> Result<Self, String> {
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
            || !valid_messages(eta)
            || fp32 && !safe_conversion(eta)
        {
            return Err("invalid CUDA topology or message conversion".into());
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

        let mut medium = vec![0u32];
        let mut wide = vec![0u32];
        for v in 0..t.n {
            let degree = t.var_offsets[2 * v + 2] - t.var_offsets[2 * v];
            if !fp32 || (33..128).contains(&degree) {
                medium.push(v as u32);
            }
            if fp32 && degree >= 128 {
                wide.push(v as u32);
            }
        }
        medium[0] = (medium.len() - 1) as u32;
        wide[0] = (wide.len() - 1) as u32;
        let mut ids = Vec::with_capacity(m + 5);
        let mut buckets = Vec::new();
        for (kind, width) in [(3, 3), (4, 4), (5, 6), (6, 12), (7, 0)] {
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
            buckets.push((kind, count, base as u32));
        }
        let real = if fp32 { 4 } else { 8 };
        let lengths = [
            t.var_offsets.len() * 4,
            edges * 4,
            t.offsets.len() * 4,
            edges * 4,
            edges * 4,
            t.n,
            m * 4,
            t.n * 4,
            edges * real,
            edges * real,
            edges * real,
            edges * real,
            edges * real,
            t.n * 2 * real,
            t.n * 4 * real,
            m * real,
            t.n * 4,
            t.n * real,
            m * real,
            32,
            4,
            medium.len() * 4,
            edges * real,
            t.n.max(m).div_ceil(256).max(1) * 4 * real,
            wide.len() * 4,
            ids.len() * 4,
            4,
            1024 * 12,
        ];
        let mut error = [0i8; 2048];
        let context = NonNull::new(unsafe { bsp_cuda_new(fp32, error.as_mut_ptr(), error.len()) })
            .ok_or_else(|| {
                unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            })?;
        let mut s = Self {
            context,
            lengths: Vec::new(),
            n: t.n,
            m,
            edges,
            fp32,
            medium: medium.len() - 1,
            wide: wide.len() - 1,
            buckets,
            assignments: vec![-1; t.n],
            patches: Vec::new(),
            profile,
            epsilon: 0.0,
            damping: 0.0,
            observables: false,
            gpu_seconds: 0.0,
            phases: Vec::new(),
            device: unsafe { CStr::from_ptr(bsp_cuda_name(context.as_ptr())) }
                .to_string_lossy()
                .into_owned(),
            allocated_bytes: lengths.iter().map(|&x| x.max(32)).sum(),
        };
        let mut free = 0;
        checked(|e, z| unsafe { bsp_cuda_budget(context.as_ptr(), &mut free, e, z) })?;
        if s.allocated_bytes > free * 3 / 4 {
            return Err("CUDA state exceeds 75% of available device memory".into());
        }
        for (slot, bytes) in lengths.into_iter().enumerate() {
            let bytes = bytes.max(32);
            checked(|e, z| unsafe { bsp_cuda_buffer(context.as_ptr(), slot as u32, bytes, e, z) })?;
            s.lengths.push(bytes);
        }
        s.write(0, t.var_offsets)?;
        s.write(1, t.var_edges)?;
        s.write(2, t.offsets)?;
        s.write(3, t.literals)?;
        s.write(4, t.edge_clause)?;
        s.write(5, &vec![-1i8; t.n])?;
        s.write(6, &vec![1u32; m])?;
        s.write(21, &medium)?;
        s.write(24, &wide)?;
        s.write(25, &ids)?;
        s.upload(eta, eta)?;
        Ok(s)
    }
    fn write<T: Copy>(&self, slot: usize, values: &[T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(values);
        if bytes == 0 {
            return Ok(());
        }
        assert!(slot < self.lengths.len() && bytes <= self.lengths[slot]);
        checked(|e, z| unsafe {
            bsp_cuda_write(
                self.context.as_ptr(),
                slot as u32,
                values.as_ptr().cast(),
                bytes,
                e,
                z,
            )
        })
    }
    fn read_into<T: Copy>(&self, slot: usize, out: &mut [T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(out);
        if bytes == 0 {
            return Ok(());
        }
        assert!(slot < self.lengths.len() && bytes <= self.lengths[slot]);
        checked(|e, z| unsafe {
            bsp_cuda_read(
                self.context.as_ptr(),
                slot as u32,
                out.as_mut_ptr().cast(),
                bytes,
                e,
                z,
            )
        })
    }
    fn real_bytes(&self) -> usize {
        if self.fp32 {
            4
        } else {
            8
        }
    }
    fn read_real(&self, slot: usize, out: &mut [f64]) -> Result<(), String> {
        if self.fp32 {
            let mut values = vec![0f32; out.len()];
            self.read_into(slot, &mut values)?;
            for (a, b) in out.iter_mut().zip(values) {
                *a = b as f64;
            }
            Ok(())
        } else {
            self.read_into(slot, out)
        }
    }
    pub fn set_assignment(&mut self, v: usize, value: i8) {
        assert!(v < self.n && (-1..=1).contains(&value));
        self.assignments[v] = value;
        self.patches.push(Patch {
            index: v as u32,
            op: 2,
            value: value as i32,
        });
    }
    pub fn set_active(&mut self, c: usize, value: bool) {
        assert!(c < self.m);
        self.patches.push(Patch {
            index: c as u32,
            op: 3,
            value: i32::from(value),
        });
    }
    pub fn freeze(&mut self, e: usize) {
        assert!(e < self.edges);
        self.patches.push(Patch {
            index: e as u32,
            op: 0,
            value: 0,
        });
    }
    pub fn restore(&mut self, e: usize) {
        assert!(e < self.edges);
        self.patches.push(Patch {
            index: e as u32,
            op: 1,
            value: 0,
        });
    }
    pub fn upload(&mut self, eta: &[f64], saved: &[f64]) -> Result<(), String> {
        if eta.len() != self.edges
            || saved.len() != self.edges
            || !valid_messages(eta)
            || !valid_messages(saved)
            || self.fp32 && (!safe_conversion(eta) || !safe_conversion(saved))
        {
            return Err("unsafe CUDA message conversion".into());
        }
        self.flush()?;
        for (slot, values) in [(8, eta), (9, saved)] {
            if self.fp32 {
                self.write(slot, &values.iter().map(|&v| v as f32).collect::<Vec<_>>())?;
            } else {
                self.write(slot, values)?;
            }
        }
        Ok(())
    }
    pub fn snapshot(&mut self, eta: &mut [f64], saved: &mut [f64]) -> Result<(), String> {
        assert_eq!(eta.len(), self.edges);
        assert_eq!(saved.len(), self.edges);
        self.flush()?;
        self.read_real(8, eta)?;
        self.read_real(9, saved)
    }
    pub fn surveys(&self, out: &mut [[f64; 3]], cert: &mut [f64]) -> Result<(), String> {
        assert_eq!(out.len(), self.n);
        assert_eq!(cert.len(), self.n);
        let mut data = vec![0.0; self.n * 4];
        self.read_real(14, &mut data)?;
        for (v, a) in data.as_chunks::<4>().0.iter().enumerate() {
            if self.assignments[v] < 0 {
                out[v] = [a[0], a[1], a[2]];
                cert[v] = a[3];
            }
        }
        Ok(())
    }
    pub fn restore_checkpoint(&mut self) -> Result<(), String> {
        checked(|e, z| unsafe {
            bsp_cuda_restore_checkpoint(self.context.as_ptr(), self.edges * self.real_bytes(), e, z)
        })
    }
    fn start(&self) -> Result<(), String> {
        checked(|e, z| unsafe { bsp_cuda_begin(self.context.as_ptr(), e, z) })
    }
    fn finish(&mut self) -> Result<f64, String> {
        let mut seconds = 0.0;
        checked(|e, z| unsafe { bsp_cuda_end(self.context.as_ptr(), &mut seconds, e, z) })?;
        self.gpu_seconds += seconds;
        Ok(seconds)
    }
    fn dispatch(&mut self, kind: usize, count: usize, base: u32) -> Result<(), String> {
        if count == 0 {
            return Ok(());
        }
        let names = [
            "cavities",
            "cavities_simd",
            "cavities_wide",
            "clauses3",
            "clauses4",
            "clauses6",
            "clauses12",
            "clauses_general",
            "residual_blocks",
            "residual_final",
            "variables",
            "clause_observables",
            "observable_blocks",
            "observable_final",
            "apply_patches",
        ];
        assert!(count <= u32::MAX as usize && kind < names.len());
        checked(|e, z| unsafe {
            bsp_cuda_dispatch(
                self.context.as_ptr(),
                kind as u32,
                count as u32,
                self.n as u32,
                self.m as u32,
                base,
                u32::from(self.observables),
                self.epsilon,
                self.damping,
                e,
                z,
            )
        })?;
        if self.profile {
            let seconds = self.finish()?;
            self.phases.push((names[kind].into(), seconds));
            self.start()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), String> {
        if self.patches.is_empty() {
            return Ok(());
        }
        self.patches
            .sort_by_key(|p| (if p.op < 2 { 0 } else { p.op - 1 }, p.index));
        let bytes = self.patches.len() * 12;
        if bytes > self.lengths[27] {
            let new = bytes.next_power_of_two();
            checked(|e, z| unsafe { bsp_cuda_buffer(self.context.as_ptr(), 27, new, e, z) })?;
            self.allocated_bytes += new - self.lengths[27];
            self.lengths[27] = new;
        }
        self.write(27, &self.patches)?;
        self.start()?;
        self.dispatch(14, self.patches.len(), 0)?;
        self.finish()?;
        self.patches.clear();
        Ok(())
    }
    fn prepare(
        &mut self,
        epsilon: f64,
        damping: f64,
        observables: bool,
        checkpoint: bool,
    ) -> Result<(), String> {
        self.epsilon = epsilon;
        self.damping = damping;
        self.observables = observables;
        self.gpu_seconds = 0.0;
        self.phases.clear();
        self.flush()?;
        self.write(
            19,
            &[Control {
                paramagnetic: 1,
                ..Control::default()
            }],
        )?;
        self.start()?;
        if checkpoint {
            checked(|e, z| unsafe {
                bsp_cuda_checkpoint(self.context.as_ptr(), self.edges * self.real_bytes(), e, z)
            })?;
        }
        Ok(())
    }
    fn cavity_dispatches(&mut self) -> Result<(), String> {
        self.dispatch(0, self.medium, 0)?;
        if self.fp32 {
            self.dispatch(1, self.n * 32, 0)?;
            self.dispatch(2, self.wide * 256, 0)?;
        }
        Ok(())
    }
    fn report(&mut self, started: Instant) -> Result<Report, String> {
        let mut control = [Control::default()];
        self.read_into(19, &mut control)?;
        let s = control[0];
        Ok(Report {
            sweeps: s.sweeps as usize,
            residual: s.residual,
            complexity: s.complexity,
            paramagnetic: s.paramagnetic != 0,
            numerical_error: s.error & 5 != 0,
            contradiction: s.error & 2 != 0,
            gpu_seconds: self.gpu_seconds,
            wall_seconds: started.elapsed().as_secs_f64(),
            phases: std::mem::take(&mut self.phases),
        })
    }
    pub fn sweeps(
        &mut self,
        count: usize,
        epsilon: f64,
        damping: f64,
        checkpoint: bool,
    ) -> Result<Report, String> {
        let started = Instant::now();
        self.prepare(epsilon, damping, false, checkpoint)?;
        for _ in 0..count {
            self.cavity_dispatches()?;
            for i in 0..self.buckets.len() {
                let (kind, count, base) = self.buckets[i];
                self.dispatch(kind, count, base)?;
            }
            self.dispatch(8, self.m, 0)?;
            self.dispatch(9, 256, 0)?;
        }
        self.finish()?;
        self.report(started)
    }
    pub fn observables(&mut self) -> Result<Report, String> {
        let started = Instant::now();
        self.prepare(0.0, 0.0, true, false)?;
        self.cavity_dispatches()?;
        self.dispatch(10, self.n, 0)?;
        self.dispatch(11, self.m, 0)?;
        self.dispatch(12, self.n.max(self.m), 0)?;
        self.dispatch(13, 1, 0)?;
        self.finish()?;
        self.report(started)
    }
}
