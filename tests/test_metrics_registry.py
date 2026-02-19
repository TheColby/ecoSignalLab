from esl.metrics.registry import create_registry


def test_builtin_registry_contains_key_metrics() -> None:
    reg = create_registry(with_external=False)
    names = set(reg.names())
    expected = {
        "rms_dbfs",
        "spl_a_db",
        "leq_db",
        "integrated_lufs",
        "snr_db",
        "spectral_centroid_hz",
        "ndsi",
        "iacc",
        "isolation_forest_score",
        "rt60_s",
        "novelty_curve",
    }
    assert expected.issubset(names)
    assert len(names) >= 74
