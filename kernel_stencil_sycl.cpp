#include <algorithm>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

std::mt19937 make_rng(unsigned int seed = std::random_device{}())
{
  return std::mt19937(seed);
}

template <typename Container>
void fillWithRandom(Container &c, std::mt19937 &gen)
{
  using T = typename Container::value_type;
  std::uniform_real_distribution<T> dis(0.0f, 1.0f);
  std::generate(std::begin(c), std::end(c), [&]()
                { return dis(gen); });
}

template <const int BM, const int BN, const int PAD, const int TM, const int TN>
sycl::event jacobi_5point_tiled_kernel(sycl::queue &q, int rows, int cols,
                                       const float *input, float *output)
{
  static_assert(BM > 0 && BN > 0, "BM and BN must be positive");
  static_assert(PAD >= 0, "PAD must be non-negative");
  static_assert(TM > 0 && TN > 0, "TM and TN must be positive");
  static_assert(BM % TM == 0, "BM must be divisible by TM");
  static_assert(BN % TN == 0, "BN must be divisible by TN");

  constexpr int local_rows = BM + 2;
  constexpr int logical_local_cols = BN + 2;
  constexpr int local_cols = logical_local_cols + PAD;
  constexpr int local_elements = local_rows * local_cols;

  const int interior_rows = rows - 2;
  const int interior_cols = cols - 2;

  sycl::range<2> local_range(BM / TM, BN / TN);
  sycl::range<2> global_range(((interior_rows + BM - 1) / BM) * (BM / TM),
                              ((interior_cols + BN - 1) / BN) * (BN / TN));

  return q.submit([&](sycl::handler &h)
                  {
    sycl::local_accessor<float, 1> tile(local_elements, h);

    h.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) {
          const int group_row = item.get_group(0);
          const int group_col = item.get_group(1);
          const int local_row_id = item.get_local_id(0);
          const int local_col_id = item.get_local_id(1);
          const int threads_per_group = item.get_local_range(0) * item.get_local_range(1);
          const int thread_id = local_row_id * item.get_local_range(1) + local_col_id;

          for (int load_idx = thread_id; load_idx < local_rows * logical_local_cols;
               load_idx += threads_per_group) {
            const int tile_row = load_idx / logical_local_cols;
            const int tile_col = load_idx % logical_local_cols;
            const int global_row = group_row * BM + tile_row;
            const int global_col = group_col * BN + tile_col;

            tile[tile_row * local_cols + tile_col] =
                (global_row < rows && global_col < cols)
                    ? input[global_row * cols + global_col]
                    : 0.0f;
          }

          sycl::group_barrier(item.get_group());

          const int first_tile_row = local_row_id * TM + 1;
          const int first_tile_col = local_col_id * TN + 1;
          const int first_global_row = group_row * BM + first_tile_row;
          const int first_global_col = group_col * BN + first_tile_col;

          for (int i = 0; i < TM; ++i) {
            const int tile_row = first_tile_row + i;
            const int global_row = first_global_row + i;

            if (global_row >= rows - 1) {
              continue;
            }

            for (int j = 0; j < TN; ++j) {
              const int tile_col = first_tile_col + j;
              const int global_col = first_global_col + j;

              if (global_col >= cols - 1) {
                continue;
              }

              const int center = tile_row * local_cols + tile_col;
              output[global_row * cols + global_col] =
                  0.2f * (tile[center] +
                          tile[center - local_cols] +
                          tile[center + local_cols] +
                          tile[center - 1] +
                          tile[center + 1]);
            }
          }
        }); });
}

// Template parameters must be defined at compile time via
// -D_BM, -D_BN, -D_BK, -D_TM, -D_TN flags.
// For this stencil kernel, _BK is local-memory row padding, not a K tile.
#ifndef _BM
#define _BM 16
#endif
#ifndef _BN
#define _BN 64
#endif
#ifndef _BK
#define _BK 0
#endif
#ifndef _TM
#define _TM 1
#endif
#ifndef _TN
#define _TN 1
#endif

extern "C" float run_kernel(int M, int N, int K, int BM_arg, int BN_arg,
                            int BK_arg, int TM_arg, int TN_arg,
                            unsigned int seed)
{
  if (BM_arg != _BM || BN_arg != _BN || BK_arg != _BK || TM_arg != _TM ||
      TN_arg != _TN)
  {
    return -1.0f;
  }

  if (M < 3 || N < 3 || K < 1)
  {
    return -1.0f;
  }

  sycl::queue q(sycl::gpu_selector_v,
                sycl::property_list{
                    sycl::property::queue::enable_profiling{},
                    sycl::property::queue::in_order{}});

  auto gen = make_rng(seed);
  std::vector<float> host_grid(M * N);
  fillWithRandom(host_grid, gen);

  float *d_current = sycl::malloc_device<float>(M * N, q);
  float *d_next = sycl::malloc_device<float>(M * N, q);
  if (!d_current || !d_next)
  {
    if (d_current)
      sycl::free(d_current, q);
    if (d_next)
      sycl::free(d_next, q);
    return -1.0f;
  }

  q.memcpy(d_current, host_grid.data(), M * N * sizeof(float));
  q.memcpy(d_next, host_grid.data(), M * N * sizeof(float));
  q.wait_and_throw();

  std::vector<sycl::event> events;
  events.reserve(K);

  for (int iter = 0; iter < K; ++iter)
  {
    sycl::event e =
        jacobi_5point_tiled_kernel<_BM, _BN, _BK, _TM, _TN>(q, M, N,
                                                            d_current, d_next);
    events.push_back(e);
    std::swap(d_current, d_next);
  }

  q.wait_and_throw();

  float elapsed_ms = 0.0f;
  for (const sycl::event &e : events)
  {
    const auto t_start =
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t_end =
        e.get_profiling_info<sycl::info::event_profiling::command_end>();
    elapsed_ms += (t_end - t_start) * 1e-6f;
  }

  sycl::free(d_current, q);
  sycl::free(d_next, q);

  return elapsed_ms;
}
