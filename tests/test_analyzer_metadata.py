from pathlib import Path
import json

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


def test_infer_spatial_metadata_exposes_hoa_channel_map_and_contract() -> None:
    meta = infer_spatial_metadata(9, "room_hoa_order2_acn_n3d.wav").to_dict()
    assert meta["layout_hint"] == "ambisonic_higher_order"
    ambi = meta["ambisonics"]
    assert ambi["order"] == 2
    assert ambi["component_order"] == "ACN"
    assert ambi["normalization"] == "N3D"
    assert ambi["standards_profile"] == "ambix_acn_n3d"
    assert ambi["normalization_scale"] == "orthonormal"
    assert ambi["channels_expected"] == 9
    assert ambi["complete_set"] is True
    assert ambi["warnings"] == []
    assert len(ambi["channel_map"]) == 9
    assert ambi["channel_map"][0] == {
        "index": 0,
        "label": "Y_0_0",
        "degree_l": 0,
        "order_m": 0,
        "acn": 0,
    }
    assert ambi["channel_map"][1]["label"] == "Y_1_-1"
    assert ambi["channel_map"][8]["label"] == "Y_2_2"


def test_infer_spatial_metadata_flags_incomplete_ambisonic_set() -> None:
    meta = infer_spatial_metadata(5, "partial_ambix_sn3d.wav").to_dict()
    assert meta["layout_family"] == "ambisonic"
    ambi = meta["ambisonics"]
    assert ambi["complete_set"] is False
    assert ambi["channels_expected"] == 9
    assert any("not a complete" in warning for warning in ambi["warnings"])


def test_analysis_applies_validated_ambisonics_sidecar(tmp_path: Path) -> None:
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    tone = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    wav = tmp_path / "recorder_channels.wav"
    sf.write(wav, np.column_stack([tone, tone, tone, tone]), sr)
    sidecar = tmp_path / "recorder_channels.spatial.json"
    sidecar.write_text(
        json.dumps(
            {
                "layout_family": "ambisonic",
                "layout_hint": "ambisonic_b_format",
                "channel_labels": ["W", "Y", "Z", "X"],
                "ambisonics": {"order": 1, "component_order": "ACN", "normalization": "SN3D"},
            }
        ),
        encoding="utf-8",
    )

    result = analyze(
        AnalysisConfig(input_path=wav, output_dir=tmp_path, verbosity=0, spatial_metadata_sidecar=sidecar)
    )
    spatial = result["metadata"]["spatial_metadata"]
    assert spatial["layout_family"] == "ambisonic"
    assert spatial["ambisonics"]["component_order"] == "ACN"
    assert spatial["provenance"]["source"] == "sidecar"
