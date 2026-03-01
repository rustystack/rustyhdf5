#!/usr/bin/env python3
"""h5py mmap-like and parallel read benchmarks."""
import time, os, tempfile, numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor

N = 1_000_000
ITERS = 50
data = np.arange(N, dtype=np.float64)

def bench(name, fn, iters=ITERS):
    for _ in range(3): fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times)//2]
    print(f"{name}: {median*1000:.3f} ms (median of {iters})")

# --- Mmap-style reads (rdcc disabled = no chunk cache, driver='core' = in-memory) ---
tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp.name, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('chunked', data=data, chunks=(10000,))
    hf.create_dataset('deflate', data=data, chunks=(10000,), compression='gzip', compression_opts=6)

def read_core_driver():
    with h5py.File(tmp.name, 'r', driver='core', backing_store=False) as hf:
        _ = hf['data'][:]

def read_default():
    with h5py.File(tmp.name, 'r') as hf:
        _ = hf['data'][:]

# --- Parallel reads: 10 datasets, read all with thread pool ---
tmp_par = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp_par.name, 'w') as hf:
    for i in range(10):
        hf.create_dataset(f'ds_{i}', data=data)

def read_10ds_sequential():
    with h5py.File(tmp_par.name, 'r') as hf:
        for i in range(10):
            _ = hf[f'ds_{i}'][:]

def read_10ds_threaded():
    def read_one(i):
        with h5py.File(tmp_par.name, 'r') as hf:
            return hf[f'ds_{i}'][:]
    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(read_one, range(10)))

# --- 100 datasets, read 1 (lazy access pattern) ---
tmp_100 = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp_100.name, 'w') as hf:
    for i in range(100):
        hf.create_dataset(f'ds_{i:04}', data=np.array([float(i)] * 10))

def read_1_of_100():
    with h5py.File(tmp_100.name, 'r') as hf:
        _ = hf['ds_0050'][:]

print("=== h5py mmap/parallel benchmarks ===\n")
bench("read_default_driver", read_default)
bench("read_core_driver (in-memory)", read_core_driver)
bench("read_10ds_sequential", read_10ds_sequential)
bench("read_10ds_threaded_10w", read_10ds_threaded, iters=20)
bench("read_1_of_100_datasets", read_1_of_100)

for f in [tmp, tmp_par, tmp_100]:
    os.unlink(f.name)
