from __future__ import annotations

import io

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def decode(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Encoded audio bytes (wav/flac/ogg/...) -> float32 mono at target_sr."""
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(data, dtype=np.float32)


def duration(audio_bytes: bytes) -> float:
    info = sf.info(io.BytesIO(audio_bytes))
    return info.frames / info.samplerate
