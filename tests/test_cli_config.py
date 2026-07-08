from pathlib import Path


def test_widar_config_exists_and_points_to_server_root():
    config = Path("configs/widar3_g6d.yaml")

    text = config.read_text(encoding="utf-8")

    assert "/home/ccl/data/csi-carat" in text
    assert "widar3g6d" in text
    assert "feature_output_dir: results/widar3_features" in text
    assert "max_steps: 20" in text
    assert "output_dir: results/widar3_erm" in text
    assert "epochs: 10" in text
    assert "run_name: multibranch" in text


def test_scripts_are_importable():
    import scripts.clean_widar3_g6d as clean_script
    import scripts.evaluate as evaluate_script
    import scripts.extract_widar3_features as feature_script
    import scripts.overfit_widar3_erm_subset as overfit_script
    import scripts.preprocess_widar3_g6d as preprocess_script
    import scripts.report_widar3_features as report_script
    import scripts.train as train_script
    import scripts.train_widar3_erm_baseline as erm_baseline_script
    import scripts.train_widar3_erm as erm_script
    import scripts.train_widar3_multibranch_erm as multibranch_script

    assert callable(clean_script.main)
    assert callable(train_script.main)
    assert callable(evaluate_script.main)
    assert callable(feature_script.main)
    assert callable(preprocess_script.main)
    assert callable(report_script.main)
    assert callable(erm_script.main)
    assert callable(erm_baseline_script.main)
    assert callable(overfit_script.main)
    assert callable(multibranch_script.main)
