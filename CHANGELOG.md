# Changelog

All notable public changes are recorded here. Versions follow the compatibility
policy in [RELEASING.md](RELEASING.md).

### CUDA

- Pack f32 survey messages into windowed buffers and run several small
  variables in each warp. On a Tesla T4, one sweep at N=10^6, K=3, alpha=4,
  seed 3 takes 3.10 ms instead of 21.83 ms (median of 9 trials). Messages stay
  bit-identical with the previous kernel. Device buffers grow by 118 MB at
  that size. The f64 path packs messages but does not use the warp kernel.
  The window size and the two-record warp are tuned on the T4.