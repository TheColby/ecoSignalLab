from __future__ import annotations

import subprocess
from pathlib import Path


def test_generate_signal_window_graphs_script(tmp_path: Path) -> None:
    out_dir = tmp_path / "figs"
    cmd = [
        ".venv/bin/python",
        "scripts/generate_signal_window_graphs.py",
        "--out",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    expected = {
        "signal_waveform.png",
        "frame_hop_overlay.png",
        "window_family.png",
        "overlap_add_hann_50pct.png",
        "spectrogram_with_frames.png",
        "spectral_flux_positive_diff.png",
        "checkerboard_kernel.png",
        "multichannel_waveforms.png",
        "foa_wxyz_waveforms.png",
        "chunk_vs_frame_timeline.png",
    }
    found = {p.name for p in out_dir.glob("*.png")}
    assert expected.issubset(found)
