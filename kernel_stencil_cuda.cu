#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
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
__global__ void jacobi_5point_tiled_kernel(int rows, int cols,
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

    __shared__ float tile[local_elements];

    const int group_row = blockIdx.x;
    const int group_col = blockIdx.y;
    const int local_row_id = threadIdx.y;
    const int local_col_id = threadIdx.x;
    const int threads_per_group = blockDim.x * blockDim.y;
    const int thread_id = local_row_id * blockDim.x + local_col_id;

    for (int load_idx = thread_id; load_idx < local_rows * logical_local_cols;
         load_idx += threads_per_group)
    {
        const int tile_row = load_idx / logical_local_cols;
        const int tile_col = load_idx % logical_local_cols;
        const int global_row = group_row * BM + tile_row;
        const int global_col = group_col * BN + tile_col;

        tile[tile_row * local_cols + tile_col] =
            (global_row < rows && global_col < cols)
                ? input[global_row * cols + global_col]
                : 0.0f;
    }

    __syncthreads();

    const int first_tile_row = local_row_id * TM + 1;
    const int first_tile_col = local_col_id * TN + 1;
    const int first_global_row = group_row * BM + first_tile_row;
    const int first_global_col = group_col * BN + first_tile_col;

    for (int i = 0; i < TM; ++i)
    {
        const int tile_row = first_tile_row + i;
        const int global_row = first_global_row + i;

        if (global_row >= rows - 1)
        {
            continue;
        }

        for (int j = 0; j < TN; ++j)
        {
            const int tile_col = first_tile_col + j;
            const int global_col = first_global_col + j;

            if (global_col >= cols - 1)
            {
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

static float cleanup_and_fail(float *d_current, float *d_next,
                              cudaEvent_t start = nullptr,
                              cudaEvent_t stop = nullptr)
{
    if (start)
        cudaEventDestroy(start);
    if (stop)
        cudaEventDestroy(stop);
    if (d_current)
        cudaFree(d_current);
    if (d_next)
        cudaFree(d_next);
    return -1.0f;
}

static float fail_cuda(const char *where, cudaError_t error, float *d_current,
                       float *d_next, cudaEvent_t start = nullptr,
                       cudaEvent_t stop = nullptr)
{
    std::fprintf(stderr, "CUDA error at %s: %s\n", where,
                 cudaGetErrorString(error));
    return cleanup_and_fail(d_current, d_next, start, stop);
}

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

    auto gen = make_rng(seed);
    std::vector<float> host_grid(M * N);
    fillWithRandom(host_grid, gen);

    float *d_current = nullptr;
    float *d_next = nullptr;
    cudaError_t error = cudaMalloc(&d_current, M * N * sizeof(float));
    if (error != cudaSuccess)
        return fail_cuda("cudaMalloc(d_current)", error, d_current, d_next);
    error = cudaMalloc(&d_next, M * N * sizeof(float));
    if (error != cudaSuccess)
        return fail_cuda("cudaMalloc(d_next)", error, d_current, d_next);
    error = cudaMemcpy(d_current, host_grid.data(), M * N * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return fail_cuda("cudaMemcpy(d_current)", error, d_current, d_next);
    error = cudaMemcpy(d_next, host_grid.data(), M * N * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return fail_cuda("cudaMemcpy(d_next)", error, d_current, d_next);

    const int interior_rows = M - 2;
    const int interior_cols = N - 2;
    dim3 dimBlock(_BN / _TN, _BM / _TM);
    dim3 dimGrid((interior_rows + _BM - 1) / _BM,
                 (interior_cols + _BN - 1) / _BN);

    cudaEvent_t start, stop;
    error = cudaEventCreate(&start);
    if (error != cudaSuccess)
        return fail_cuda("cudaEventCreate(start)", error, d_current, d_next);
    error = cudaEventCreate(&stop);
    if (error != cudaSuccess)
        return fail_cuda("cudaEventCreate(stop)", error, d_current, d_next,
                         start);

    error = cudaEventRecord(start);
    if (error != cudaSuccess)
        return fail_cuda("cudaEventRecord(start)", error, d_current, d_next,
                         start, stop);
    for (int iter = 0; iter < K; ++iter)
    {
        jacobi_5point_tiled_kernel<_BM, _BN, _BK, _TM, _TN>
            <<<dimGrid, dimBlock>>>(M, N, d_current, d_next);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return fail_cuda("kernel launch", error, d_current, d_next, start,
                             stop);
        std::swap(d_current, d_next);
    }
    error = cudaEventRecord(stop);
    if (error != cudaSuccess)
        return fail_cuda("cudaEventRecord(stop)", error, d_current, d_next,
                         start, stop);
    error = cudaEventSynchronize(stop);
    if (error != cudaSuccess)
        return fail_cuda("cudaEventSynchronize(stop)", error, d_current,
                         d_next, start, stop);

    float elapsed_ms = 0.0f;
    error = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (error != cudaSuccess)
        return fail_cuda("cudaEventElapsedTime", error, d_current, d_next,
                         start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_current);
    cudaFree(d_next);

    return elapsed_ms;
}
