from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.core import AnalysisConfig, analyze


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe are required for this test")
def test_ffmpeg_decoder_provenance_for_wma(tmp_path: Path) -> None:
    sr = 16_000
    t = np.linspace(0.0, 0.5, int(0.5 * sr), endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    wav = tmp_path / "tone.wav"
    wma = tmp_path / "tone.wma"
    sf.write(wav, x, sr)

    encode = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav),
            "-c:a",
            "wmav2",
            str(wma),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert encode.returncode == 0, encode.stderr
    assert wma.exists()

    result = analyze(
        AnalysisConfig(
            input_path=wma,
            output_dir=tmp_path,
            metrics=["rms_dbfs", "snr_db"],
            verbosity=0,
        )
    )

    decoder = result["metadata"]["decoder"]  # type: ignore[index]
    assert decoder["decoder_used"] == "ffmpeg"  # type: ignore[index]
    assert isinstance(decoder.get("ffmpeg_version"), str) and decoder["ffmpeg_version"]  # type: ignore[index]

    ffprobe = decoder.get("ffprobe")
    assert isinstance(ffprobe, dict)
    assert isinstance(ffprobe.get("codec_name"), str)
    assert int(ffprobe.get("channels", 0)) == 1
    assert int(ffprobe.get("sample_rate", 0)) > 0
    duration = ffprobe.get("duration_s")
    assert duration is None or float(duration) > 0.0
