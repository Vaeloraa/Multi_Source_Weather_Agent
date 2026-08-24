# Remote SEVIR VIL reading

The local data layer now uses this fallback order:

```text
local HDF5 -> remote S3 HDF5 -> per-event NPZ cache
```

`CATALOG.csv` still supplies the event metadata, HDF5 file name, and row index. When the indexed HDF5 file is not present under `data/sevir_complete/data`, the loader opens the public SEVIR object through `fsspec` HTTP range reads, extracts one `vil[file_index]` event, and stores it under `data/sevir_event_cache/`.

The default remote prefix is:

```text
https://sevir.s3.us-west-2.amazonaws.com/data
```

Environment variables:

```powershell
$env:SEVIR_REMOTE_ENABLED = "1"
$env:SEVIR_REMOTE_BASE_URL = "https://sevir.s3.us-west-2.amazonaws.com/data"
$env:SEVIR_REMOTE_BLOCK_SIZE = "8388608"
```

Set `SEVIR_REMOTE_ENABLED=0` to restore the previous local-only behavior. The remote cache is intentionally ignored by Git.

CasCast live inference uses the same loader. For a remote event, it materializes only the selected 25-frame segment into a temporary NPZ (`13` history frames plus `12` Ground Truth frames), runs the existing worker, and removes that temporary input after inference. The final CasCast output remains in the existing live-inference cache.

The first verified remote read loaded event `S696651` in about 5 seconds and produced a `(49, 384, 384)` array. A subsequent read from the per-event cache took about 0.05 seconds and used a cache file of about 1.67 MB.
