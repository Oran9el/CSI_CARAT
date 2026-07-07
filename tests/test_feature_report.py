from pathlib import Path
import pickle

import numpy as np

from csi_carat.data.feature_report import summarize_feature_cache, write_feature_report


def _write_feature_cache(path: Path) -> None:
    payload = {
        "amplitude": np.ones((2, 3, 4), dtype=np.float32),
        "phase_difference": np.zeros((2, 3, 4), dtype=np.float32),
        "doppler_spectrogram": np.ones((2, 3, 5, 2), dtype=np.float32) * 2,
        "activities": np.array([1, 2], dtype=np.int8),
        "environments": np.array([1, 2], dtype=np.int8),
        "users": np.array([5, 1], dtype=np.int8),
        "domains": np.array([0, 9], dtype=np.int8),
        "source_indices": np.array([10, 20], dtype=np.int32),
        "window_starts": np.array([0, 64], dtype=np.int32),
        "feature_config": {"n_fft": 8},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_summarize_feature_cache_reports_shapes_and_label_counts(tmp_path):
    cache_path = tmp_path / "features.pkl"
    _write_feature_cache(cache_path)

    summary = summarize_feature_cache(cache_path)

    assert summary["num_windows"] == 2
    assert summary["branches"]["amplitude"]["shape"] == [2, 3, 4]
    assert summary["branches"]["amplitude"]["dtype"] == "float32"
    assert summary["branches"]["amplitude"]["finite"] is True
    assert summary["activity_counts"] == {1: 1, 2: 1}
    assert summary["domain_counts"] == {0: 1, 9: 1}


def test_write_feature_report_creates_markdown(tmp_path):
    cache_path = tmp_path / "features.pkl"
    report_path = tmp_path / "report.md"
    _write_feature_cache(cache_path)
    summary = summarize_feature_cache(cache_path)

    write_feature_report(summary, report_path)

    text = report_path.read_text(encoding="utf-8")
    assert "# Widar3 Feature Cache Report" in text
    assert "amplitude" in text
    assert "num_windows: 2" in text
