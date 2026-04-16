from pathlib import Path

import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.core.spatial_metadata import infer_spatial_metadata


def test_analyzer_emits_runtime_metadata(tmp_path: Path) -> None:
    sr = 8000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, x, sr)

    result = analyze(AnalysisConfig(input_path=wav, output_dir=tmp_path, verbosity=0))

    assert "analysis_time_local" in result
    assert isinstance(result.get("schema_version"), str)
    assert isinstance(result.get("pipeline_hash"), str)
    assert isinstance(result.get("metric_catalog"), dict)
    assert isinstance(result.get("library_versions"), dict)
    meta = result["metadata"]
    assert meta.get("channel_layout_hint") in {"mono", "stereo", "multichannel", "ambisonic_b_format"}
    assert isinstance(meta.get("spatial_metadata"), dict)
    assert meta["spatial_metadata"]["layout_hint"] == meta.get("channel_layout_hint")
    assert isinstance(meta.get("runtime"), dict)
    assert "python" in meta["runtime"]
    assert isinstance(meta.get("decoder"), dict)
    assert "decoder_used" in meta["decoder"]
    assert isinstance(meta.get("config_snapshot"), dict)
    assert isinstance(meta.get("resolved_metric_list"), list)
    assert isinstance(meta.get("metric_catalog_version"), str)
    assert isinstance(meta.get("channel_metrics"), dict)
    assert isinstance(meta.get("validity_flags"), dict)


def test_infer_spatial_metadata_recognizes_ambix_foa() -> None:
    meta = infer_spatial_metadata(4, "forest_ambix_sn3d.wav").to_dict()
    assert meta["layout_hint"] == "ambisonic_b_format"
    ambi = meta["ambisonics"]
    assert ambi["component_order"] == "ACN"
    assert ambi["normalization"] == "SN3D"
    assert ambi["order"] == 1
