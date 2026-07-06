from pathlib import Path


def test_widar_config_exists_and_points_to_server_root():
    config = Path("configs/widar3_g6d.yaml")

    text = config.read_text(encoding="utf-8")

    assert "/home/ccl/data/csi-carat" in text
    assert "widar3g6d" in text


def test_scripts_are_importable():
    import scripts.evaluate as evaluate_script
    import scripts.train as train_script

    assert callable(train_script.main)
    assert callable(evaluate_script.main)
