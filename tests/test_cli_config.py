from pathlib import Path


def test_widar_config_exists_and_points_to_server_root():
    config = Path("configs/widar3_g6d.yaml")

    text = config.read_text(encoding="utf-8")

    assert "/home/ccl/data/csi-carat" in text
    assert "widar3g6d" in text


def test_scripts_are_importable():
    import scripts.clean_widar3_g6d as clean_script
    import scripts.evaluate as evaluate_script
    import scripts.extract_widar3_features as feature_script
    import scripts.preprocess_widar3_g6d as preprocess_script
    import scripts.train as train_script

    assert callable(clean_script.main)
    assert callable(train_script.main)
    assert callable(evaluate_script.main)
    assert callable(feature_script.main)
    assert callable(preprocess_script.main)
