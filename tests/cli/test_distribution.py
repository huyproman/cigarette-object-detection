"""Installable resources must not depend on a source checkout."""
from importlib.resources import files
import json


def test_registry_is_a_packaged_resource():
    registry = files("p05_detect").joinpath("configs/registry/model_registry.json")
    assert json.loads(registry.read_text())["default_profile"] == "b0"


def test_model_definitions_are_packaged_resources():
    from p05_detect.engine.model import CONFIG_DIR

    assert CONFIG_DIR.is_relative_to(files("p05_detect"))
    for name in ("yolo26.yaml", "yolo26s-carafe.yaml", "yolo26s-p2.yaml", "yolo26s-mhsa.yaml"):
        assert (CONFIG_DIR / name).is_file()
