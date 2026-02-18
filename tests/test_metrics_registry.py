from esl.metrics.registry import create_registry


def test_builtin_registry_contains_key_metrics() -> None:
    reg = create_registry(with_external=False)
    names = set(reg.names())
    expected = {
        "rms_dbfs",
        "spl_a_db",
        "snr_db",
        "spectral_centroid_hz",
        "rt60_s",
        "novelty_curve",
    }
    assert expected.issubset(names)
    assert len(names) >= 20
