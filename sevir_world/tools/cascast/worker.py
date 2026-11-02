from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


SEGMENT_STARTS = (0, 12, 24)
LATENT_SCALE = 0.6786020398139954


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CasCast locally from an official SEVIR VIL HDF5 event.")
    parser.add_argument("--cascast-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hdf5", type=Path)
    source.add_argument("--input-npz", type=Path)
    parser.add_argument("--event-id")
    parser.add_argument("--hdf5-row", type=int)
    parser.add_argument("--source-id")
    parser.add_argument("--segment", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--ensemble-members", type=int, default=1)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--stage", choices=("earthformer", "full"), default="full")
    parser.add_argument(
        "--rolling-segments", type=int, choices=(1, 2, 3), default=1,
        help="For an uploaded sequence, recursively forecast this many 60-minute segments.",
    )
    parser.add_argument(
        "--cache-request-key", default="",
        help="versioned artifact identity supplied by the service",
    )
    parser.add_argument("--pipeline-version", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ensemble_members < 1 or args.ensemble_members > 10:
        parser.error("--ensemble-members must be between 1 and 10")
    if args.hdf5 is not None and not args.event_id:
        parser.error("--event-id is required with --hdf5")
    if args.input_npz is not None and not args.source_id:
        parser.error("--source-id is required with --input-npz")
    return args


def _checkpoint_state(path: Path, submodel: str) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model"][submodel]
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state.items()
    )


def _read_event(
    path: Path,
    event_id: str,
    segment: int,
    hdf5_row: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    encoded_id = event_id.encode("ascii")
    with h5py.File(path, "r") as handle:
        ids = handle["id"][:]
        if hdf5_row is not None:
            row = int(hdf5_row)
            if not 0 <= row < len(ids):
                raise IndexError(f"HDF5 row {row} is outside {path.name}")
            if ids[row] != encoded_id:
                raise ValueError(
                    f"HDF5 row {row} contains {ids[row]!r}, not event {event_id!r}"
                )
        else:
            matches = np.flatnonzero(ids == encoded_id)
            if len(matches) != 1:
                raise ValueError(
                    f"event {event_id!r} occurs {len(matches)} times in {path.name}; "
                    "an exact --hdf5-row is required"
                )
            row = int(matches[0])
        event = np.asarray(handle["vil"][row], dtype=np.uint8)
    if event.shape != (384, 384, 49):
        raise ValueError(f"unexpected VIL event shape: {event.shape}")
    start = SEGMENT_STARTS[segment]
    sequence = np.moveaxis(event[:, :, start : start + 25], -1, 0)
    return sequence[:13], sequence[13:], row


def _read_uploaded_sequence(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        history = np.asarray(payload["pixels"], dtype=np.uint8)
        ground_truth = (
            np.asarray(payload["ground_truth"], dtype=np.uint8)
            if "ground_truth" in payload
            else None
        )
    if history.shape != (13, 384, 384):
        raise ValueError(f"unexpected uploaded sequence shape: {history.shape}")
    if ground_truth is not None and ground_truth.shape != (12, 384, 384):
        raise ValueError(f"unexpected input ground_truth shape: {ground_truth.shape}")
    return history, ground_truth


def _free_cuda(*models: torch.nn.Module) -> None:
    for model in models:
        model.to("cpu")
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_earthformer(root: Path) -> torch.nn.Module:
    from networks.earthformer_xy import EarthFormer_xy

    model = EarthFormer_xy(in_len=13, out_len=12, height=384, width=384)
    checkpoint = root / "experiments" / "EarthFormer" / "world_size1-ckpt" / "earthformer_ckpt.pth"
    model.load_state_dict(_checkpoint_state(checkpoint, "EarthFormer_xy"), strict=True)
    return model.eval()


def _load_autoencoder(root: Path) -> torch.nn.Module:
    from networks.autoencoder_kl import autoencoder_kl

    config = {
        "in_channels": 1,
        "out_channels": 1,
        "down_block_types": ["DownEncoderBlock2D"] * 4,
        "up_block_types": ["UpDecoderBlock2D"] * 4,
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2,
        "latent_channels": 4,
        "norm_num_groups": 32,
    }
    model = autoencoder_kl(config=config)
    checkpoint = root / "ckpts" / "autoencoder" / "ckpt.pth"
    model.load_state_dict(_checkpoint_state(checkpoint, "autoencoder_kl"), strict=True)
    return model.eval()


def _load_casformer(root: Path) -> torch.nn.Module:
    from networks.casformer import CasFormer

    model = CasFormer(
        arch="DiT-custom",
        config={
            "input_size": 48,
            "in_channels": 8,
            "mlp_ratio": 4.0,
            "learn_sigma": False,
            "out_channels": 48,
            "split_num": 12,
            "num_heads": 16,
            "single_heads_num": 4,
            "hidden_size": 1152,
            "enc_hidden_size": 256,
            "patch_size": 2,
            "enc_depth": 12,
            "latent_depth": 12,
        },
    )
    checkpoint = root / "experiments" / "cascast_diffusion" / "world_size1-ckpt" / "checkpoint_best.pth"
    model.load_state_dict(_checkpoint_state(checkpoint, "casformer"), strict=True)
    return model.eval()


@torch.inference_mode()
def _earthformer_prediction(history: np.ndarray, root: Path, device: torch.device) -> np.ndarray:
    # EarthFormer's cuboid attention is numerically unstable in fp16 for this checkpoint.
    model = _load_earthformer(root).to(device=device, dtype=torch.float32)
    inputs = torch.from_numpy(history.astype(np.float32) / 255.0)[None, :, None].to(device=device)
    prediction = model(inputs)
    result = prediction.float().cpu().numpy()[0, :, 0]
    del prediction, inputs
    _free_cuda(model)
    return result


@torch.inference_mode()
def _encode_prediction(prediction: np.ndarray, root: Path, device: torch.device, dtype: torch.dtype, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = _load_autoencoder(root).to(device=device, dtype=dtype)
    outputs: list[torch.Tensor] = []
    for start in range(0, 12, 2):
        batch = torch.from_numpy(prediction[start : start + 2, None]).to(device=device, dtype=dtype)
        posterior = model.net.encode(batch)
        outputs.append(posterior.sample().float().cpu())
        del batch, posterior
    latent = torch.cat(outputs).numpy()
    _free_cuda(model)
    return latent


@torch.inference_mode()
def _sample_latents(
    latent: np.ndarray,
    root: Path,
    device: torch.device,
    dtype: torch.dtype,
    members: int,
    guidance: float,
    seed: int,
) -> np.ndarray:
    from src.diffusers import DDIMScheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample_range=13,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(20, device=device)
    model = _load_casformer(root).to(device=device, dtype=dtype)
    condition = torch.from_numpy(latent)[None].to(device=device, dtype=dtype) * LATENT_SCALE
    cfg_condition = torch.cat((condition, torch.zeros_like(condition)))
    sampled: list[torch.Tensor] = []
    for member in range(members):
        generator = torch.Generator(device=device).manual_seed(seed + member)
        value = torch.randn(
            condition.shape, generator=generator, device=device, dtype=dtype
        ) * scheduler.init_noise_sigma
        for timestep in scheduler.timesteps:
            model_input = torch.cat((value, value))
            model_input = scheduler.scale_model_input(model_input, timestep)
            timesteps = torch.full((2,), float(timestep), device=device, dtype=dtype)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.float16):
                noise = model(x=model_input, timesteps=timesteps, cond=cfg_condition)
            conditional, unconditional = noise.chunk(2)
            noise = unconditional + guidance * (conditional - unconditional)
            value = scheduler.step(noise, timestep, value).prev_sample
        sampled.append(value.float().cpu())
    result = torch.cat(sampled).numpy()
    del condition, cfg_condition
    _free_cuda(model)
    return result


@torch.inference_mode()
def _decode_latents(latents: np.ndarray, root: Path, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    model = _load_autoencoder(root).to(device=device, dtype=dtype)
    decoded_members: list[np.ndarray] = []
    for member in latents:
        frames: list[torch.Tensor] = []
        for start in range(0, 12, 2):
            batch = torch.from_numpy(member[start : start + 2] / LATENT_SCALE).to(device=device, dtype=dtype)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.float16):
                decoded = model.net.decode(batch)
            frames.append(decoded.float().cpu())
            del batch, decoded
        decoded_members.append(torch.cat(frames).numpy()[:, 0] * 255.0)
    _free_cuda(model)
    return np.stack(decoded_members).astype(np.float32)


def _cascast_block(
    history: np.ndarray,
    root: Path,
    device: torch.device,
    dtype: torch.dtype,
    ensemble_members: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one 60-minute block and return members plus their mean."""
    deterministic = _earthformer_prediction(history, root, device)
    if not np.isfinite(deterministic).all():
        raise RuntimeError("EarthFormer produced non-finite values")
    latent = _encode_prediction(deterministic, root, device, dtype, seed)
    sampled = _sample_latents(
        latent, root, device, dtype, ensemble_members, 2.0, seed
    )
    members = _decode_latents(sampled, root, device, dtype)
    return members, members.mean(axis=0, dtype=np.float32)


def main() -> None:
    args = _arguments()
    root = args.cascast_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local CasCast inference")
    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    started = time.time()
    if args.hdf5 is not None:
        history, ground_truth, row = _read_event(
            args.hdf5, args.event_id, args.segment, args.hdf5_row
        )
        source_id = args.source_id or args.event_id
    else:
        history, ground_truth = _read_uploaded_sequence(args.input_npz)
        row = -1
        source_id = args.source_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "earthformer":
        deterministic = _earthformer_prediction(history, root, device)
        if not np.isfinite(deterministic).all():
            raise RuntimeError("EarthFormer produced non-finite values")
        np.savez_compressed(
            args.output,
            source_id=np.asarray(source_id),
            hdf5_row=np.asarray(row),
            segment=np.asarray(args.segment),
            cache_request_key=np.asarray(args.cache_request_key),
            pipeline_version=np.asarray(args.pipeline_version),
            history=history,
            earthformer_prediction=deterministic,
        )
        print(json.dumps({"status": "ok", "stage": "earthformer", "output": str(args.output), "seconds": round(time.time() - started, 2)}))
        return
    if args.rolling_segments > 1:
        # Recursive conditioning uses the previous block's ensemble mean as the
        # next 13-frame input. This is a single coherent event trajectory and
        # avoids concatenating unrelated events; later-member spread remains
        # conditional on the mean trajectory rather than independent branches.
        current_history = history.copy()
        blocks: list[np.ndarray] = []
        deterministic = np.asarray([])
        for segment in range(args.rolling_segments):
            block_members, block_mean = _cascast_block(
                current_history, root, device, dtype,
                args.ensemble_members, args.seed + segment * args.ensemble_members,
            )
            blocks.append(block_members)
            current_history = np.concatenate(
                [current_history[-1:], np.clip(block_mean, 0, 255)], axis=0
            ).astype(np.uint8)
        members = np.concatenate(blocks, axis=1)
    else:
        deterministic = _earthformer_prediction(history, root, device)
        if not np.isfinite(deterministic).all():
            raise RuntimeError("EarthFormer produced non-finite values")
        latent = _encode_prediction(deterministic, root, device, dtype, args.seed)
        sampled = _sample_latents(
            latent, root, device, dtype, args.ensemble_members, args.guidance, args.seed
        )
        members = _decode_latents(sampled, root, device, dtype)
    mean = members.mean(axis=0, dtype=np.float32)
    # For a local HDF5 event, retain the complete future truth corresponding
    # to all recursively forecast blocks (not only the first 60 minutes).
    if args.hdf5 is not None and args.rolling_segments > 1:
        with h5py.File(args.hdf5, "r") as handle:
            full_event = np.asarray(handle["vil"][row], dtype=np.uint8)
        start = SEGMENT_STARTS[args.segment]
        truth_end = min(start + 13 + 12 * args.rolling_segments, full_event.shape[-1])
        ground_truth = np.moveaxis(full_event[:, :, start + 13:truth_end], -1, 0)
    np.savez_compressed(
        args.output,
        source_id=np.asarray(source_id),
        hdf5_row=np.asarray(row),
        segment=np.asarray(args.segment),
        history=history,
        earthformer_prediction=deterministic,
        ensemble_members=members,
        ensemble_mean=mean,
        pixel_std=members.std(axis=0, dtype=np.float32),
        q10=np.quantile(members, 0.10, axis=0).astype(np.float32),
        q90=np.quantile(members, 0.90, axis=0).astype(np.float32),
        member_seed=np.arange(args.seed, args.seed + args.ensemble_members, dtype=np.int64),
        rolling_segments=np.asarray(args.rolling_segments),
        cache_request_key=np.asarray(args.cache_request_key),
        pipeline_version=np.asarray(args.pipeline_version),
        ground_truth_available=np.asarray(ground_truth is not None),
        **({"ground_truth": ground_truth} if ground_truth is not None else {}),
    )
    print(json.dumps({"status": "ok", "stage": "full", "output": str(args.output), "seconds": round(time.time() - started, 2)}))


if __name__ == "__main__":
    main()
