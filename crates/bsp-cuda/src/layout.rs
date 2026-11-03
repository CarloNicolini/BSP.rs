//! Host index layout for the packed f32 sweep.
//!
//! The windows and the warp records match the bit-identical T4 kernel: eta is
//! gathered inside source windows of 2^18 edges, and q is stored inside
//! position windows of 2^16 edges.

const ETA_WINDOW: u32 = 1 << 18;
const Q_WINDOW: u32 = 1 << 16;

pub struct Packed {
    pub src: Vec<u32>,
    pub eta_at: Vec<u32>,
    pub q_slot: Vec<u32>,
    pub q_at: Vec<u32>,
    pub quads: Vec<u32>,
    pub positions: Vec<u32>,
    pub clause_ids: Vec<u32>,
    pub buckets: Vec<(usize, usize, u32)>,
    pub simd_lanes: usize,
}

pub fn pack(
    var_offsets: &[u32],
    var_edges: &[u32],
    offsets: &[u32],
    clause_ids: &[u32],
    buckets: Vec<(usize, usize, u32)>,
    fp32: bool,
) -> Packed {
    let (src, eta_at, q_slot, q_at) = windows(var_edges);
    let (quads, positions) = if fp32 {
        warp_plan(var_offsets)
    } else {
        (Vec::new(), Vec::new())
    };
    let simd_lanes = positions.len() * 32;
    let (clause_ids, buckets) = clause_headers(offsets, clause_ids, buckets);
    Packed {
        src,
        eta_at,
        q_slot,
        q_at,
        quads,
        positions,
        clause_ids,
        buckets,
        simd_lanes,
    }
}

fn windows(var_edges: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let edges = var_edges.len();
    let mut order: Vec<u32> = (0..edges as u32).collect();
    order.sort_by_key(|&p| var_edges[p as usize] / ETA_WINDOW);
    let mut src = vec![0u32; edges];
    let mut eta_at = vec![0u32; edges];
    for (k, &p) in order.iter().enumerate() {
        src[k] = var_edges[p as usize];
        eta_at[p as usize] = k as u32;
    }
    let mut position = vec![0u32; edges];
    for (p, &e) in var_edges.iter().enumerate() {
        position[e as usize] = p as u32;
    }
    order.clear();
    order.extend(0..edges as u32);
    order.sort_by_key(|&e| position[e as usize] / Q_WINDOW);
    let mut q_at = vec![0u32; edges];
    for (k, &e) in order.iter().enumerate() {
        q_at[e as usize] = k as u32;
    }
    let q_slot = var_edges.iter().map(|&e| q_at[e as usize]).collect();
    (src, eta_at, q_slot, q_at)
}

fn clause_headers(
    offsets: &[u32],
    old: &[u32],
    buckets: Vec<(usize, usize, u32)>,
) -> (Vec<u32>, Vec<(usize, usize, u32)>) {
    let mut ids = Vec::new();
    let mut out = Vec::with_capacity(buckets.len());
    for (kind, count, base) in buckets {
        let base = if (3..=7).contains(&kind) {
            let start = base as usize + 1;
            let members = &old[start..start + count];
            let new_base = ids.len() as u32;
            let contiguous =
                kind < 7 && count > 0 && members[count - 1] - members[0] == count as u32 - 1;
            if contiguous {
                ids.push(count as u32 | 1 << 31);
                ids.push(members[0]);
                ids.push(offsets[members[0] as usize]);
            } else {
                ids.push(count as u32);
                ids.extend_from_slice(members);
            }
            new_base
        } else {
            base
        };
        out.push((kind, count, base));
    }
    (ids, out)
}

fn warp_plan(var_offsets: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = var_offsets.len() / 2;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut degree = vec![0u32; n];
    let mut positives = vec![0u32; n];
    for v in 0..n {
        degree[v] = var_offsets[2 * v + 2] - var_offsets[2 * v];
        positives[v] = var_offsets[2 * v + 1] - var_offsets[2 * v];
    }
    let mut warp_of = vec![0u32; n];
    let mut offset_of = vec![0u32; n];
    let mut warp = 0u32;
    let mut offset = 0u32;
    for v in 0..n {
        let small = degree[v] <= 32;
        let width = if !small || degree[v] < 2 {
            1
        } else {
            degree[v].next_power_of_two()
        };
        offset = offset.div_ceil(width) * width;
        if offset + width > 32 {
            warp += 1;
            offset = 0;
        }
        warp_of[v] = warp;
        offset_of[v] = offset;
        offset = if small { offset + width } else { 32 };
    }
    let nwarps = warp as usize + 1;
    let mut first = vec![0u32; nwarps];
    let mut starts = vec![0u32; nwarps];
    let mut edges = vec![0u32; nwarps];
    let mut positive = vec![0u32; nwarps];
    let mut seen = vec![false; nwarps];
    for v in 0..n {
        let w = warp_of[v] as usize;
        if !seen[w] {
            seen[w] = true;
            first[w] = v as u32;
        }
        let off = offset_of[v];
        starts[w] |= 1u32 << off;
        if degree[v] <= 32 {
            edges[w] |= mask(degree[v], off);
            positive[w] |= mask(positives[v], off);
        } else {
            positive[w] |= 1u32 << off;
        }
    }
    let mut quads = Vec::with_capacity(nwarps * 4);
    let mut positions = Vec::with_capacity(nwarps);
    for w in 0..nwarps {
        quads.extend([first[w], starts[w], edges[w], positive[w]]);
        positions.push(var_offsets[2 * first[w] as usize]);
    }
    (quads, positions)
}

fn mask(bits: u32, offset: u32) -> u32 {
    if bits == 0 {
        0
    } else if bits == 32 {
        u32::MAX
    } else {
        ((1u32 << bits) - 1) << offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_records_match_the_packed_schedule() {
        let offsets = [0, 0, 1, 2, 4, 12, 20, 20, 60, 64, 68];
        let (quads, positions) = warp_plan(&offsets);
        assert_eq!(
            quads,
            [0, 65553, 4294901873, 16711696, 3, 1, 0, 1, 4, 1, 255, 15]
        );
        assert_eq!(positions, [0, 20, 60]);
    }

    #[test]
    fn one_window_keeps_variable_order() {
        let edges: Vec<u32> = (0..8).collect();
        let (src, eta_at, q_slot, q_at) = windows(&edges);
        assert_eq!(src, edges);
        assert_eq!(eta_at, edges);
        assert_eq!(q_at, edges);
        assert_eq!(q_slot, edges);
    }

    #[test]
    fn contiguous_clauses_use_a_header() {
        let offsets = [0, 3, 6, 10];
        let ids = [3u32, 0, 1, 2];
        let (ids, buckets) = clause_headers(&offsets, &ids, vec![(3, 3, 0)]);
        assert_eq!(ids, [3 | 1 << 31, 0, 0]);
        assert_eq!(buckets, [(3, 3, 0)]);
    }
}
