"""Payload and probability-format estimates, not measured peak memory.

The selected layout is the norm/prefix reuse experiment on the development
branch. Its device savings apply to Metal; CUDA still uses the baseline layout.
Run from the repository root with Python 3.12 or newer.
"""
import json

from memory_budget import budget


def main():
    workloads = []
    for k, alpha in [(3, 4.2), (4, 4.2), (4, 9.8)]:
        scales = []
        for n in [10**6, 10**7, 10**8, 10**9]:
            base = budget(n, k, alpha)
            e = base["e"]
            gpu = base["current_gpu_f32_main_buffers_u32_layout"]
            host = base["current_layout_total"]
            scales.append(dict(n=n, m=base["m"], edges=e,
                baseline_scalar_payload=host,
                selected_scalar_payload=host - 8 * e,
                selected_parallel_payload=host - 8 * e + base["parallel_scratch_addition_u32_layout"],
                baseline_device_payload=gpu,
                selected_metal_payload=gpu - 8 * e + 8,
                one_eta_fp64=8 * e, one_eta_fp32=4 * e,
                one_eta_16bit=2 * e,
                hypothetical_implicit_fp32=base["implicit_proposals"]["32"]["total"],
                hypothetical_implicit_16bit=base["implicit_proposals"]["16"]["total"],
                current_u32_graph_accepts=base["current_u32_accepts"],
                current_gpu_dispatch_accepts=n <= (2**32 - 1)//32
                    and e < 2**32 - 1 and base["m"] <= 2**32 - 1 - 255))
        workloads.append(dict(k=k, alpha=alpha, incidences_per_variable=k*alpha,
            u32_incidence_n_bound_exclusive=(2**32 - 1)/(k*alpha), scales=scales))
    formats = []
    for name, fraction, emin in [("fp64",52,-1022), ("fp32",23,-126),
                                  ("bf16",7,-126), ("fp16",10,-14)]:
        # Spacing below one is 2^(-fraction-1); half that rounds to one.
        formats.append(dict(name=name, fraction_bits=fraction,
            unit_roundoff=2.0**(-fraction-1),
            max_absolute_rounding_error_on_unit_interval=2.0**(-fraction-2),
            rounds_to_one_within=2.0**(-fraction-2),
            min_normal=2.0**emin, min_subnormal=2.0**(emin-fraction),
            factors_of_half_at_min_normal=-emin,
            factors_of_half_at_min_subnormal=fraction-emin,
            k3_equal_q_at_min_normal=2.0**(emin/2),
            k4_equal_q_at_min_normal=2.0**(emin/3),
            k3_equal_q_at_min_subnormal=2.0**((emin-fraction)/2),
            k4_equal_q_at_min_subnormal=2.0**((emin-fraction)/3)))
    print(json.dumps(dict(workloads=workloads, formats=formats,
        note="u32 layouts extrapolated for comparison beyond their valid index limits; "
             "exclude allocation/construction/driver peaks and small bucket lists. "
             "GPU state is additional to the host state. Implicit profiles are proposals, "
             "not implementations or validated numerical formats."), indent=2))


if __name__ == "__main__":
    main()
