"""Resource guards. Import this FIRST, before numpy, in every entry point.

Caps BLAS/OpenMP threads before numpy loads, so a matrix op cannot saturate every core. Lives beside
the entry points rather than inside the `unitsim` package: a library must not rewrite its caller's
environment variables on import.

Set UNITSIM_THREADS to override the cap (default 4, leaving headroom).
"""
import os


def _default_threads():
    n = os.cpu_count() or 4
    return str(max(1, min(4, n - 2)))


_T = os.environ.get("UNITSIM_THREADS", _default_threads())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

THREADS = int(_T)


def available_ram_bytes():
    """Physical RAM currently available, without requiring psutil."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullAvailPhys)
    except Exception:
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 1024 ** 3


def report(prefix=""):
    print(f"{prefix}threads={THREADS}  free RAM={available_ram_bytes() / 1e9:.2f} GB", flush=True)
