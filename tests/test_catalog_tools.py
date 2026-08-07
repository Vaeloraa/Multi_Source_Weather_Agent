from __future__ import annotations

import csv
import base64
import tempfile
import unittest
from pathlib import Path

from sevir_world.backends import geospatial, statistics
from sevir_world.data_access import catalog as tools
from sevir_world.data_access import uploads
import h5py
import numpy as np


FIELDS = [
    "id", "file_name", "file_index", "img_type", "time_utc", "minute_offsets",
    "episode_id", "event_id", "event_type", "llcrnrlat", "llcrnrlon",
    "urcrnrlat", "urcrnrlon", "proj", "size_x", "size_y", "height_m",
    "width_m", "data_min", "data_max", "pct_missing",
]


class CatalogToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.catalog = root / "CATALOG.csv"
        self.data_root = root / "data"
        h5_path = self.data_root / "vil" / "test.h5"
        h5_path.parent.mkdir(parents=True)

        values = np.zeros((2, 4, 4, 49), dtype=np.uint8)
        for frame in range(49):
            values[0, :, :, frame] = frame
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("vil", data=values)

        base = {
            "file_name": "vil/test.h5",
            "file_index": "0",
            "img_type": "vil",
            "time_utc": "2019-11-06 06:42:05",
            "minute_offsets": ":".join(str(value) for value in range(-120, 125, 5)),
            "episode_id": "",
            "event_id": "",
            "event_type": "",
            "llcrnrlat": "41.68553947335055",
            "llcrnrlon": "-93.25478150329442",
            "urcrnrlat": "44.84805067102971",
            "urcrnrlon": "-88.1205048846413",
            "proj": "+proj=laea +lat_0=38 +lon_0=-98 +units=m +a=6370997.0 +ellps=sphere",
            "size_x": "4",
            "size_y": "4",
            "height_m": "384000.0",
            "width_m": "384000.0",
            "data_min": "0",
            "data_max": "48",
            "pct_missing": "0",
        }
        rows = [
            {**base, "id": "R_TEST"},
            {
                **base,
                "id": "S_TEST",
                "file_index": "1",
                "episode_id": "115911.0",
                "event_id": "696651.0",
                "event_type": "Hail",
            },
        ]
        with self.catalog.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        self.original_catalog = tools.CATALOG_CSV
        self.original_data_root = tools.LOCAL_SEVIR_DATA_ROOT
        tools.CATALOG_CSV = self.catalog
        tools.LOCAL_SEVIR_DATA_ROOT = self.data_root
        self._clear_catalog_caches()

    def tearDown(self) -> None:
        tools.CATALOG_CSV = self.original_catalog
        tools.LOCAL_SEVIR_DATA_ROOT = self.original_data_root
        self._clear_catalog_caches()
        self.temp.cleanup()

    @staticmethod
    def _clear_catalog_caches() -> None:
        tools._catalog_vil_records.cache_clear()
        tools._catalog_vil_index.cache_clear()

    def test_metadata_restores_strict_frame_times(self) -> None:
        record = tools.get_sevir_event_metadata("R_TEST")["record"]
        self.assertEqual(len(record["minute_offsets"]), 49)
        self.assertEqual(record["frame_times_utc"][0], "2019-11-06T04:42:05Z")
        self.assertEqual(record["frame_times_utc"][-1], "2019-11-06T08:42:05Z")

    def test_hdf5_loader_uses_file_index_and_time_axis(self) -> None:
        result = tools.load_sevir_vil_event("R_TEST", start_frame=12, end_frame=15)
        self.assertEqual(result["full_event_shape"], [49, 4, 4])
        self.assertEqual(result["loaded_shape"], [3, 4, 4])
        self.assertEqual(result["frame_statistics"][0]["max_vil"], 12)
        self.assertEqual(result["frame_statistics"][-1]["max_vil"], 14)

    def test_projection_and_noaa_linkage(self) -> None:
        location = geospatial.geolocate_sevir_event("R_TEST", row=2, column=2)["pixel_location"]
        self.assertEqual(location["pixel_resolution_m"], [96000.0, 96000.0])
        reference = statistics.get_noaa_storm_reference("S_TEST")
        self.assertTrue(reference["noaa_linked"])
        self.assertEqual(reference["noaa_event_id"], 696651)
        self.assertIn("id=696651", reference["noaa_event_url"])


class UploadPersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.original_upload_dir = uploads.UPLOAD_STORAGE_DIR
        uploads.UPLOAD_STORAGE_DIR = Path(self.temp.name) / "uploads"
        uploads.UPLOADED_VIL.clear()
        uploads.UPLOADED_VIL_SEQUENCES.clear()

    def tearDown(self) -> None:
        uploads.UPLOAD_STORAGE_DIR = self.original_upload_dir
        uploads.UPLOADED_VIL.clear()
        uploads.UPLOADED_VIL_SEQUENCES.clear()
        self.temp.cleanup()

    def test_single_upload_survives_memory_clear(self) -> None:
        frame = np.arange(384 * 384, dtype=np.uint32).astype(np.uint8).reshape(384, 384)
        result = uploads.register_uploaded_vil(
            filename="sample.png",
            width=384,
            height=384,
            pixels_base64=base64.b64encode(frame.tobytes()).decode("ascii"),
            grayscale_source=True,
        )
        upload_id = result["upload_id"]
        uploads.UPLOADED_VIL.clear()

        restored, metadata = uploads._load_uploaded_vil(upload_id)
        self.assertTrue(np.array_equal(restored, frame))
        self.assertEqual(metadata["filename"], "sample.png")
        records = uploads.list_uploaded_records()
        self.assertEqual(records["record_count"], 1)
        self.assertEqual(records["records"][0]["upload_id"], upload_id)

    def test_sequence_can_be_restored_from_disk(self) -> None:
        upload_id = "a" * 32
        sequence = np.zeros((13, 384, 384), dtype=np.uint8)
        metadata = {
            "upload_id": upload_id,
            "kind": "sequence",
            "filenames": [f"frame_{index:02d}.png" for index in range(13)],
            "shape": [13, 384, 384],
            "dtype": "uint8",
            "created_at_utc": "2026-08-07T00:00:00+00:00",
        }
        uploads._persist_uploaded_array(upload_id, sequence, metadata)
        restored, restored_metadata = uploads._load_uploaded_vil_sequence(upload_id)
        self.assertEqual(restored.shape, (13, 384, 384))
        self.assertEqual(restored_metadata["kind"], "sequence")


if __name__ == "__main__":
    unittest.main()
