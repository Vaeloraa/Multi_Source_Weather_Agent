# Local CasCast inference

The live pipeline reads one official SEVIR VIL event from a local HDF5 file and runs:

```text
13 observed VIL frames
-> EarthFormer deterministic 12-frame forecast
-> Autoencoder latent encoding
-> CasCast DDIM diffusion (20 steps)
-> Autoencoder decoding
-> raw VIL prediction + aligned 12-frame ground truth
```

## Web UI

1. Start the existing server normally.
2. Select `本地HDF5现场推理` as the data source.
3. Select an event. The list only contains events backed by a downloaded HDF5 file.
4. Ask for a forecast or hazard analysis and run the Agent.

For user data, select `上传13帧并现场预测`, upload thirteen chronological
384x384 grayscale VIL PNG files, and run the analysis. This path uses the same
local model chain but has no forecast metrics unless twelve future ground-truth
frames are supplied separately.

The first one-member request normally takes tens of seconds on the local GPU. The result is cached under `data/live_cascast_cache`; repeated requests reuse that result.

## Direct verification

```powershell
$env:PYTHONPATH = (Resolve-Path ".\vendor").Path
@'
from sevir_world.backends.forecast import run_nowcast

result = run_nowcast(
    event_id="R19082500047459",
    model_name="cascast_live",
    threshold=133,
    input_frames=13,
    forecast_frames=12,
)
print(result["inference_mode"], result["model_metrics"])
'@ | python -
```

Raw decoded output is retained for evaluation and may overshoot the nominal VIL display range. Display and threshold analysis use a clipped engineering view; the raw cache is not overwritten.
