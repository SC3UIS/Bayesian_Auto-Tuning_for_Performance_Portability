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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
sycl::event sgemm_blocktiling_2d_kernel(sycl::queue &q, int num_rows_a,
                                        int num_cols_b, int num_cols_a,
                                        float alpha, const float *matrix_a,
                                        const float *matrix_b, float beta,
                                        float *matrix_c)
{

  sycl::range<2> local_range(BM / TM, BN / TN);
  sycl::range<2> global_range(((num_rows_a + BM - 1) / BM) * (BM / TM),
                              ((num_cols_b + BN - 1) / BN) * (BN / TN));

  return q.submit([&](sycl::handler &h)
                  {
    sycl::local_accessor<float, 1> tile_a(BM * BK, h);
    sycl::local_accessor<float, 1> tile_b(BK * BN, h);

    h.parallel_for(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) {
          const uint block_row = item.get_group(0);
          const uint block_col = item.get_group(1);
          const uint thread_row = item.get_local_id(0);
          const uint thread_col = item.get_local_id(1);
          const uint tid = thread_row * (BN / TN) + thread_col;

          const int global_row = block_row * BM + thread_row * TM;
          const int global_col = block_col * BN + thread_col * TN;

          int a_offset = block_row * BM * num_cols_a;
          int b_offset = block_col * BN;
          int c_offset = block_row * BM * num_cols_b + block_col * BN;

          float thread_results[TM * TN] = {0.0f};
          float register_m[TM] = {0.0f};
          float register_n[TN] = {0.0f};

          for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BK) {
            for (uint load_offset = 0; load_offset < BM * BK;
                 load_offset += (BM / TM) * (BN / TN)) {
              uint load_idx = tid + load_offset;
              if (load_idx < BM * BK) {
                uint a_row = load_idx / BK;
                uint a_col = load_idx % BK;
                tile_a[load_idx] =
                    ((block_row * BM + a_row) < num_rows_a &&
                     (tile_idx + a_col) < num_cols_a)
                        ? matrix_a[a_offset + a_row * num_cols_a + a_col]
                        : 0.0f;
              }
            }

            for (uint load_offset = 0; load_offset < BK * BN; load_offset += (BM / TM) * (BN / TN)) {
              uint load_idx = tid + load_offset;
              if (load_idx < BK * BN) {
                uint b_row = load_idx / BN;
                uint b_col = load_idx % BN;
                tile_b[load_idx] =
                    ((tile_idx + b_row) < num_cols_a &&
                     (block_col * BN + b_col) < num_cols_b)
                        ? matrix_b[b_offset + b_row * num_cols_b + b_col]
                        : 0.0f;
              }
            }

            sycl::group_barrier(item.get_group());

            a_offset += BK;
            b_offset += BK * num_cols_b;

            for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
              for (uint i = 0; i < TM; ++i) {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
              }

              for (uint i = 0; i < TN; ++i) {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
              }

              for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                  thread_results[res_idx_m * TN + res_idx_n] +=
                      register_m[res_idx_m] * register_n[res_idx_n];
                }
              }
            }

            sycl::group_barrier(item.get_group());
          }

          for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
            for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
              int row = global_row + res_idx_m;
              int col = global_col + res_idx_n;
              if (row < num_rows_a && col < num_cols_b) {
                const int c_idx = c_offset +
                                  (thread_row * TM + res_idx_m) * num_cols_b +
                                  thread_col * TN + res_idx_n;
                matrix_c[c_idx] =
                    alpha * thread_results[res_idx_m * TN + res_idx_n] +
                    beta * matrix_c[c_idx];
              }
            }
          }
        }); });
}

// Template parameters must be defined at compile time via -D_BM, -D_BN, -D_BK, -D_TM, -D_TN flags
#ifndef _BM
#define _BM 32
#endif
#ifndef _BN
#define _BN 32
#endif
#ifndef _BK
#define _BK 32
#endif
#ifndef _TM
#define _TM 1
#endif
#ifndef _TN
#define _TN 1
#endif

extern "C" float run_kernel(int M, int N, int K, int BM_arg, int BN_arg, int BK_arg, int TM_arg, int TN_arg,
                            unsigned int seed)
{
  if (BM_arg != _BM || BN_arg != _BN || BK_arg != _BK || TM_arg != _TM || TN_arg != _TN)
  {
    return -1.0f; // Parameter mismatch
  }

  sycl::queue q(sycl::gpu_selector_v,
                sycl::property::queue::enable_profiling{});

  auto gen = make_rng(seed);
  std::vector<float> A(M * K), B(K * N);
  fillWithRandom(A, gen);
  fillWithRandom(B, gen);

  float *dA = sycl::malloc_device<float>(M * K, q);
  float *dB = sycl::malloc_device<float>(K * N, q);
  float *dC = sycl::malloc_device<float>(M * N, q);
  if (!dA || !dB || !dC) {
    if (dA)
      sycl::free(dA, q);
    if (dB)
      sycl::free(dB, q);
    if (dC)
      sycl::free(dC, q);
    return -1.0f;
  }

  q.memcpy(dA, A.data(), M * K * sizeof(float));
  q.memcpy(dB, B.data(), K * N * sizeof(float));
  q.memset(dC, 0, M * N * sizeof(float));
  q.wait_and_throw();

  const float alpha = 1.0f, beta = 0.0f;

  sycl::event e = sgemm_blocktiling_2d_kernel<_BM, _BN, _BK, _TM, _TN>(q, M, N, K, alpha, dA, dB, beta, dC);
  e.wait_and_throw();

  auto t_start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto t_end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);

  return (t_end - t_start) * 1e-6f;
}
