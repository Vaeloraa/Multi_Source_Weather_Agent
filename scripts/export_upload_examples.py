"""Export a few upload-ready grayscale PNGs from the local official SEVIR event."""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "sevir_sample" / "sevir_vil_event.npz"
OUTPUT = ROOT / "web" / "examples"


def main() -> None:
    event = np.load(SOURCE)
    vil = event["vil"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for frame_index in (12, 24, 36):
        image = Image.fromarray(vil[:, :, frame_index].astype(np.uint8), mode="L")
        image.save(OUTPUT / f"S696651_frame_{frame_index + 1:02d}.png")


if __name__ == "__main__":
    main()
