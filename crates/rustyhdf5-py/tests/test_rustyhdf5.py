"""Tests for rustyhdf5 Python bindings."""

import os
import tempfile

import numpy as np
import pytest

import rustyhdf5


@pytest.fixture
def tmp_h5(tmp_path):
    """Return a temporary HDF5 file path."""
    return str(tmp_path / "test.h5")


@pytest.fixture
def sample_read_file(tmp_h5):
    """Create a sample HDF5 file for reading tests."""
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("temperatures", data=np.array([22.5, 23.1, 21.8]))
        f.create_dataset("counts", data=np.array([10, 20, 30], dtype=np.int32))
        f.attrs["version"] = 1
        f.attrs["description"] = "test file"
    return tmp_h5


@pytest.fixture
def grouped_read_file(tmp_h5):
    """Create an HDF5 file with groups for reading tests."""
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("root_data", data=np.array([0.0, 1.0]))
        grp = f.create_group("sensors")
        grp.create_dataset("temperature", data=np.array([22.5, 23.1, 21.8]))
        grp.create_dataset("humidity", data=np.array([45, 50, 55], dtype=np.int32))
        grp.attrs["location"] = "lab"
        grp2 = f.create_group("metadata")
        grp2.create_dataset("timestamps", data=np.array([1000, 2000, 3000], dtype=np.int64))
    return tmp_h5


# ---------------------------------------------------------------------------
# Test: open and read datasets
# ---------------------------------------------------------------------------


def test_open_and_read_f64(sample_read_file):
    f = rustyhdf5.File(sample_read_file, "r")
    ds = f["temperatures"]
    data = ds[:]
    np.testing.assert_array_almost_equal(data, [22.5, 23.1, 21.8])
    f.close()


def test_open_and_read_i32(sample_read_file):
    f = rustyhdf5.File(sample_read_file, "r")
    ds = f["counts"]
    data = ds[:]
    np.testing.assert_array_equal(data, [10, 20, 30])
    assert data.dtype == np.int32
    f.close()


# ---------------------------------------------------------------------------
# Test: dataset properties (shape, dtype)
# ---------------------------------------------------------------------------


def test_dataset_shape(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        ds = f["temperatures"]
        assert ds.shape == (3,)


def test_dataset_dtype(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        assert f["temperatures"].dtype == "float64"
        assert f["counts"].dtype == "int32"


# ---------------------------------------------------------------------------
# Test: read attributes
# ---------------------------------------------------------------------------


def test_read_root_attrs(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        assert f.attrs["version"] == 1
        assert f.attrs["description"] == "test file"


def test_attrs_len(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        assert len(f.attrs) >= 2


def test_attrs_contains(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        assert "version" in f.attrs
        assert "nonexistent" not in f.attrs


def test_attrs_keys(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        keys = f.attrs.keys()
        assert "version" in keys
        assert "description" in keys


# ---------------------------------------------------------------------------
# Test: read groups
# ---------------------------------------------------------------------------


def test_read_group_keys(grouped_read_file):
    with rustyhdf5.File(grouped_read_file, "r") as f:
        keys = f.keys()
        assert "sensors" in keys
        assert "metadata" in keys
        assert "root_data" in keys


def test_read_group_dataset(grouped_read_file):
    with rustyhdf5.File(grouped_read_file, "r") as f:
        grp = f["sensors"]
        ds = grp["temperature"]
        data = ds[:]
        np.testing.assert_array_almost_equal(data, [22.5, 23.1, 21.8])


def test_read_group_attrs(grouped_read_file):
    with rustyhdf5.File(grouped_read_file, "r") as f:
        grp = f["sensors"]
        assert grp.attrs["location"] == "lab"


def test_nested_path_access(grouped_read_file):
    """Test f['group/dataset'] path navigation."""
    with rustyhdf5.File(grouped_read_file, "r") as f:
        ds = f["sensors/temperature"]
        data = ds[:]
        np.testing.assert_array_almost_equal(data, [22.5, 23.1, 21.8])


# ---------------------------------------------------------------------------
# Test: context manager
# ---------------------------------------------------------------------------


def test_context_manager(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        data = f["temperatures"][:]
        np.testing.assert_array_almost_equal(data, [22.5, 23.1, 21.8])
    # File should be closed after with block
    assert repr(f) == "<HDF5 File (closed)>"


# ---------------------------------------------------------------------------
# Test: create files / write mode
# ---------------------------------------------------------------------------


def test_write_simple(tmp_h5):
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("data", data=np.array([1.0, 2.0, 3.0]))
    # Verify by reading back
    with rustyhdf5.File(tmp_h5, "r") as f:
        data = f["data"][:]
        np.testing.assert_array_almost_equal(data, [1.0, 2.0, 3.0])


def test_write_with_attrs(tmp_h5):
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("values", data=np.array([10, 20], dtype=np.int32))
        f.attrs["author"] = "test"
        f.attrs["count"] = 42
    with rustyhdf5.File(tmp_h5, "r") as f:
        assert f.attrs["author"] == "test"
        assert f.attrs["count"] == 42


def test_write_with_group(tmp_h5):
    with rustyhdf5.File(tmp_h5, "w") as f:
        grp = f.create_group("experiment")
        grp.create_dataset("results", data=np.array([3.14, 2.72]))
        grp.attrs["version"] = 1
    with rustyhdf5.File(tmp_h5, "r") as f:
        ds = f["experiment/results"]
        np.testing.assert_array_almost_equal(ds[:], [3.14, 2.72])
        grp = f["experiment"]
        assert grp.attrs["version"] == 1


# ---------------------------------------------------------------------------
# Test: numpy array types round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_float64(tmp_h5):
    original = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("data", data=original)
    with rustyhdf5.File(tmp_h5, "r") as f:
        result = f["data"][:]
        np.testing.assert_array_almost_equal(result, original)
        assert result.dtype == np.float64


def test_roundtrip_float32(tmp_h5):
    original = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("data", data=original)
    with rustyhdf5.File(tmp_h5, "r") as f:
        result = f["data"][:]
        np.testing.assert_array_almost_equal(result, original)
        assert result.dtype == np.float32


def test_roundtrip_int32(tmp_h5):
    original = np.array([-10, 0, 10, 100], dtype=np.int32)
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("data", data=original)
    with rustyhdf5.File(tmp_h5, "r") as f:
        result = f["data"][:]
        np.testing.assert_array_equal(result, original)
        assert result.dtype == np.int32


def test_roundtrip_int64(tmp_h5):
    original = np.array([-1, 0, 1, 2**40], dtype=np.int64)
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("data", data=original)
    with rustyhdf5.File(tmp_h5, "r") as f:
        result = f["data"][:]
        np.testing.assert_array_equal(result, original)
        assert result.dtype == np.int64


def test_roundtrip_uint8(tmp_h5):
    original = np.array([0, 127, 255], dtype=np.uint8)
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("data", data=original)
    with rustyhdf5.File(tmp_h5, "r") as f:
        result = f["data"][:]
        np.testing.assert_array_equal(result, original)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test: chunked + compressed datasets
# ---------------------------------------------------------------------------


def test_chunked_gzip(tmp_h5):
    original = np.arange(100, dtype=np.float64)
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset(
            "compressed",
            data=original,
            chunks=(50,),
            compression="gzip",
            compression_opts=6,
        )
    with rustyhdf5.File(tmp_h5, "r") as f:
        result = f["compressed"][:]
        np.testing.assert_array_equal(result, original)


# ---------------------------------------------------------------------------
# Test: h5py interoperability
# ---------------------------------------------------------------------------


def test_h5py_can_read_our_file(tmp_h5):
    """Verify that h5py can read files we create."""
    import h5py

    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("values", data=np.array([1.0, 2.0, 3.0]))
        f.attrs["meta"] = "hello"
    with h5py.File(tmp_h5, "r") as f:
        np.testing.assert_array_equal(f["values"][:], [1.0, 2.0, 3.0])
        # h5py reads fixed-length strings as bytes
        assert f.attrs["meta"] == b"hello"


def test_we_can_read_h5py_file(tmp_h5):
    """Verify that we can read files created by h5py."""
    import h5py

    with h5py.File(tmp_h5, "w") as f:
        f.create_dataset("data", data=np.array([10.0, 20.0, 30.0]))
        f.attrs["version"] = 2
    with rustyhdf5.File(tmp_h5, "r") as f:
        data = f["data"][:]
        np.testing.assert_array_equal(data, [10.0, 20.0, 30.0])
        assert f.attrs["version"] == 2


# ---------------------------------------------------------------------------
# Test: 2D array shape
# ---------------------------------------------------------------------------


def test_2d_array_roundtrip(tmp_h5):
    original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    with rustyhdf5.File(tmp_h5, "w") as f:
        f.create_dataset("matrix", data=original)
    with rustyhdf5.File(tmp_h5, "r") as f:
        ds = f["matrix"]
        assert ds.shape == (2, 3)
        result = ds[:]
        np.testing.assert_array_almost_equal(result, original)


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------


def test_open_nonexistent_file():
    with pytest.raises(OSError):
        rustyhdf5.File("/nonexistent/path.h5", "r")


def test_invalid_mode(tmp_h5):
    with pytest.raises(ValueError):
        rustyhdf5.File(tmp_h5, "x")


def test_key_error_on_missing_dataset(sample_read_file):
    with rustyhdf5.File(sample_read_file, "r") as f:
        with pytest.raises(KeyError):
            f["nonexistent"]
