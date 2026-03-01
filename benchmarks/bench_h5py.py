#!/usr/bin/env python3
"""h5py benchmarks matching rustyhdf5 Criterion benches (1M f64)."""
import time, os, tempfile, numpy as np, h5py

N = 1_000_000
ITERS = 50
data = np.arange(N, dtype=np.float64)

def bench(name, fn, iters=ITERS):
    # warmup
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times)//2]
    print(f"{name}: {median*1000:.3f} ms (median of {iters})")

# --- Write benchmarks ---
def write_contiguous():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as f:
        with h5py.File(f.name, 'w') as hf:
            hf.create_dataset('data', data=data)

def write_chunked():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as f:
        with h5py.File(f.name, 'w') as hf:
            hf.create_dataset('data', data=data, chunks=(10000,))

def write_chunked_deflate():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as f:
        with h5py.File(f.name, 'w') as hf:
            hf.create_dataset('data', data=data, chunks=(10000,), compression='gzip', compression_opts=6)

# --- Read benchmarks (pre-create files) ---
tmp_contig = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp_contig.name, 'w') as hf:
    hf.create_dataset('data', data=data)

tmp_chunked = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp_chunked.name, 'w') as hf:
    hf.create_dataset('data', data=data, chunks=(10000,))

tmp_deflate = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp_deflate.name, 'w') as hf:
    hf.create_dataset('data', data=data, chunks=(10000,), compression='gzip', compression_opts=6)

def read_contiguous():
    with h5py.File(tmp_contig.name, 'r') as hf:
        _ = hf['data'][:]

def read_chunked():
    with h5py.File(tmp_chunked.name, 'r') as hf:
        _ = hf['data'][:]

def read_chunked_deflate():
    with h5py.File(tmp_deflate.name, 'r') as hf:
        _ = hf['data'][:]

# --- Metadata: parse superblock + read attrs ---
tmp_attrs = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp_attrs.name, 'w') as hf:
    for i in range(50):
        hf.attrs[f'attr_{i}'] = f'value_{i}'

def read_50_attrs():
    with h5py.File(tmp_attrs.name, 'r') as hf:
        for i in range(50):
            _ = hf.attrs[f'attr_{i}']

# --- Group navigation (100 groups) ---
tmp_groups = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
with h5py.File(tmp_groups.name, 'w') as hf:
    g = hf
    for i in range(100):
        g = g.create_group(f'g{i}')
    g.create_dataset('leaf', data=[1.0])

def group_nav_100():
    with h5py.File(tmp_groups.name, 'r') as hf:
        path = '/'.join(f'g{i}' for i in range(100)) + '/leaf'
        _ = hf[path][:]

print(f"=== h5py {h5py.version.version} / HDF5 {h5py.version.hdf5_version} / numpy {np.__version__} ===")
print(f"=== {N:,} float64 elements ({N*8/1e6:.1f} MB) ===\n")

bench("write_contiguous", write_contiguous)
bench("write_chunked", write_chunked)
bench("write_chunked_deflate", write_chunked_deflate, iters=20)
bench("read_contiguous", read_contiguous)
bench("read_chunked", read_chunked)
bench("read_chunked_deflate", read_chunked_deflate)
bench("read_50_attrs", read_50_attrs)
bench("group_nav_100", group_nav_100)

# cleanup
for f in [tmp_contig, tmp_chunked, tmp_deflate, tmp_attrs, tmp_groups]:
    os.unlink(f.name)
