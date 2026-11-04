// Stable-sort reference from dc6483d, retained only for comparison.
const ETA_WINDOW: u32 = 1 << 18;
const Q_WINDOW: u32 = 1 << 16;
pub fn windows(var_edges: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
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
