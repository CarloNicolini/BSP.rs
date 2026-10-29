"""Analytical payload budgets; no allocation and no claim of peak RSS.

The current-layout model intentionally extrapolates beyond current u32 limits.
Proposed implicit profiles include explicit workspace allowances, not measured
implementations. All byte counts use N variables, M=floor(alpha*N), E=k*M.
"""

import argparse
import json
import math


def budget(n, k, alpha):
    m = math.floor(alpha * n)
    e = k * m
    words = m * math.ceil(k / 64)
    current = {
        "formula": 12 * e + 8 * (m + 1) + 4 * (2 * n + 1) + 8 * words,
        "solver_edge_arrays": 32 * e,
        "solver_variable_arrays_and_reserved_ids": 66 * n + 8 * math.ceil(n / 64),
        "solver_clause_arrays": 8 * words + 12 * m + 8 * math.ceil(m / 64),
    }
    # u64 edge references, clause offsets, signed-variable CSR offsets, free_xor.
    # Clause IDs and mask offsets still fit u32 only when M and words fit u32.
    widening = 4 * e + 4 * (m + 1) + 4 * (2 * n + 1) + 4 * m
    parallel = 32 * e + 24 * n + 8 * m
    walksat = 5 * n + 16 * m
    # Fixed k=3/4: free-slot mask + satisfied-literal count fits one byte.
    clause_bytes = math.ceil((k + math.ceil(math.log2(k + 1))) / 8)
    implicit = {}
    for bits in (32, 16, 8):
        components = {
            "one_message_payload": bits // 8 * e,
            "two_f64_variable_aggregates": 16 * n,
            "two_u8_exact_zero_counts": 2 * n,
            "f32_frozen_decision_scores": 4 * n,
            "packed_variable_state_and_scan_bitset": math.ceil(n / 2),
            "clause_state_and_scan_bitset": clause_bytes * m + 8 * math.ceil(m / 64),
            "unary_degree_boundaries_without_rank_select_index": math.ceil((e + n) / 8),
            "rank_select_and_exception_allowance": math.ceil(n / 2),
            "bounded_queues_tiles_selection_allowance": 2 * n,
        }
        implicit[str(bits)] = {"components": components, "total": sum(components.values())}
    return {
        "n": n, "k": k, "alpha": alpha, "m": m, "e": e,
        "current_u32_accepts": e < 2**32 - 1 and m < 2**32 - 1,
        "current_layout_components": current,
        "current_layout_total": sum(current.values()),
        "minimal_edge_widening_addition_when_clause_ids_fit": widening if m < 2**32 - 1 and words < 2**32 - 1 else None,
        "parallel_scratch_addition_u32_layout": parallel,
        "walksat_peak_addition_u32_layout": walksat,
        # Main CUDA/Metal f32 buffers. Degree-bucket IDs and tiny constants omitted.
        "current_gpu_f32_main_buffers_u32_layout": 36 * e + 45 * n + 20 * m + 16 * math.ceil(max(n, m) / 256),
        "implicit_proposals": implicit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10**9)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=4.2)
    args = parser.parse_args()
    if args.n < 1 or args.k < 1 or not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("require positive N, k, and finite alpha")
    print(json.dumps(budget(args.n, args.k, args.alpha), indent=2))


if __name__ == "__main__":
    main()
