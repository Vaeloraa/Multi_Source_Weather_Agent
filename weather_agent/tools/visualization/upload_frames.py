"""Select raw uploaded frames for the generic visualization renderer."""

from __future__ import annotations

import numpy as np

from ...request.uploads import _load_uploaded_vil, _load_uploaded_vil_sequence


def uploaded_frame(upload_id: str) -> np.ndarray:
    """Return one uploaded field without applying a source-specific style."""
    frame, _ = _load_uploaded_vil(upload_id)
    return np.asarray(frame, dtype=np.float32)


def uploaded_sequence_frame(
    upload_id: str,
    frame_index: int,
) -> np.ndarray:
    """Select one raw frame from an uploaded sequence for input preview."""
    sequence, _ = _load_uploaded_vil_sequence(upload_id)
    if not 0 <= int(frame_index) < sequence.shape[0]:
        raise IndexError("uploaded sequence frame index is outside the available range")
    return np.asarray(sequence[int(frame_index)], dtype=np.float32)
