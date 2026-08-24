from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sevir_world.data_access import catalog


class _Context:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, traceback):
        return False


class _Dataset:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape

    def __getitem__(self, index):
        return self.values[index]


class _Handle(dict):
    def __contains__(self, key):
        return key in self.keys()


class RemoteSevirCacheTests(unittest.TestCase):
    def test_remote_event_is_cached_after_first_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_root = Path(temporary)
            values = np.zeros((1, 384, 384, 49), dtype=np.uint8)
            values[0, 10, 20, 3] = 249
            handle = _Handle(vil=_Dataset(values))
            record = {
                "id": "RTEST00000001",
                "file_name": "vil/2019/test.h5",
                "file_index": 0,
                "minute_offsets": tuple(range(-240, 5, 5)),
            }
            with (
                patch.object(catalog, "SEVIR_EVENT_CACHE", cache_root),
                patch.object(catalog, "_local_hdf5_path", return_value=cache_root / "missing.h5"),
                patch.object(catalog.fsspec, "open", return_value=_Context(object())),
                patch.object(catalog.h5py, "File", return_value=_Context(handle)) as h5_file,
            ):
                first, source = catalog._load_vil_array(record)
                second, cached_source = catalog._load_vil_array(record)

            self.assertEqual(first.shape, (49, 384, 384))
            self.assertEqual(first[3, 10, 20], 249)
            np.testing.assert_array_equal(first, second)
            self.assertTrue(source.startswith("remote:"))
            self.assertTrue(cached_source.startswith("cache:"))
            h5_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
