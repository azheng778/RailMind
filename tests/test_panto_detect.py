import os

import numpy as np
import pytest

from railmind.capabilities.panto import detect


class FakeDetector:
    mode = "yolo"
    error = None
    weights_path = "fake.pt"

    def __init__(self, detections):
        self.detections = detections

    def predict(self, image, conf=0.4):
        return self.detections


def test_deployed_weights_are_the_default(monkeypatch):
    monkeypatch.delenv("RAILMIND_PANTO_WEIGHTS", raising=False)
    assert detect.resolve_weights_path() == detect.DEPLOYED_WEIGHTS
    assert os.path.isfile(detect.resolve_weights_path())


def test_environment_override_takes_priority(monkeypatch, tmp_path):
    custom = tmp_path / "custom.pt"
    monkeypatch.setenv("RAILMIND_PANTO_WEIGHTS", str(custom))
    assert detect.resolve_weights_path() == str(custom.resolve())


def test_detect_parts_keeps_supported_classes_and_sorts(monkeypatch):
    fake = FakeDetector([
        {"class_id": 2, "confidence": 0.7, "bbox_xyxy": [10, 10, 40, 40]},
        {"class_id": 9, "confidence": 0.99, "bbox_xyxy": [0, 0, 10, 10]},
        {"class_id": 0, "confidence": 0.9, "bbox_xyxy": [2, 2, 12, 12]},
    ])
    monkeypatch.setattr(detect, "get_model", lambda: fake)
    result = detect.detect_parts(np.zeros((64, 64, 3), dtype=np.uint8))
    assert [item["class_id"] for item in result] == [0, 2]


def test_crop_strip_clamps_bounds_and_prefers_useful_region(monkeypatch):
    detections = [
        {"class_id": 2, "confidence": 0.95, "bbox_xyxy": [1, 1, 15, 15]},
        {"class_id": 2, "confidence": 0.80, "bbox_xyxy": [-5, 10, 80, 55]},
    ]
    monkeypatch.setattr(detect, "get_model", lambda: FakeDetector(detections))
    image = np.zeros((60, 70, 3), dtype=np.uint8)
    crop = detect.crop_strip(image, pad_ratio=0.2)
    assert crop is not None
    assert crop.shape[1] == 70
    assert 45 <= crop.shape[0] <= 60


@pytest.mark.parametrize("conf", [0, -0.1, 1.1])
def test_detect_parts_rejects_invalid_confidence(conf):
    with pytest.raises(ValueError, match="E_INVALID_CONF"):
        detect.detect_parts(np.zeros((32, 32, 3), dtype=np.uint8), conf=conf)
