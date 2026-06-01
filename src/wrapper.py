import ctypes

_ARGTYPES = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint
]
_RESTYPE = ctypes.c_float

_lib_paths = {
    ("sycl", "matmul"): "./kernel_matmul_sycl.so",
    ("cuda", "matmul"): "./kernel_matmul_cuda.so",
    ("sycl", "stencil"): "./kernel_stencil_sycl.so",
    ("cuda", "stencil"): "./kernel_stencil_cuda.so",
}
_libs = {}

def _load(path):
    lib = ctypes.CDLL(path)
    lib.run_kernel.argtypes = _ARGTYPES
    lib.run_kernel.restype  = _RESTYPE
    return lib                          

def _get_lib(backend, kernel):
    key = (backend, kernel)
    if key not in _lib_paths:           
        raise ValueError(
            f"Unknown backend/kernel '{backend}/{kernel}'. "
            f"Choose from: {list(_lib_paths)}"
        )
    if key not in _libs:
        _libs[key] = _load(_lib_paths[key])
    return _libs[key]                   

def run(M, N, K, BM, BN, BK, TM, TN=1, backend="sycl", kernel="matmul", seed=42):
    lib = _get_lib(backend, kernel)
    return lib.run_kernel(M, N, K, BM, BN, BK, TM, TN, seed)
