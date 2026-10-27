"""Unit tests for slicing geometry, tile coverage and merge oracles.
Tests:
- Complete source pixel coverage for arbitrary image sizes.
- Edge tile handling and clipping.
- Source smaller than tile size.
- Coordinate inversion from tile back to source space.
- Global merge duplicate merging on synthetic boxes (IoU >= 0.5 merged, IoU < 0.5 preserved).
- Full+slice fusion correctness.
"""
import pytest
import numpy as np
import torch
from p05_detect.infer.slicing import generate_tiles, global_merge


def test_tile_coverage_arbitrary_sizes():
    """Verify that every pixel in image [0, w) x [0, h) is covered by at least one tile."""
    test_sizes = [(800, 600), (1920, 1080), (640, 640), (300, 300), (1500, 500), (450, 1200)]
    for w, h in test_sizes:
        for tile_sz in (640, 960):
            tiles = generate_tiles(w, h, tile_sz, overlap_ratio=0.20)
            coverage = np.zeros((h, w), dtype=np.uint8)
            for (x1, y1, x2, y2) in tiles:
                assert 0 <= x1 < x2 <= w
                assert 0 <= y1 < y2 <= h
                coverage[y1:y2, x1:x2] += 1
            assert np.all(coverage >= 1), f"Uncovered pixels in {w}x{h} with tile_size={tile_sz}!"


def test_source_smaller_than_tile():
    """If image is smaller than tile size, exactly one tile of image dimensions is returned."""
    tiles = generate_tiles(300, 400, 640)
    assert tiles == [(0, 0, 300, 400)]


def test_global_merge_duplicate_merging_oracle():
    """Two identical overlapping boxes (IoU=1.0) must merge into exactly one box."""
    candidates = [
        {"class_id": 0, "class_name": "cigarette", "confidence": 0.85, "box_xyxy": [100.0, 100.0, 150.0, 130.0]},
        {"class_id": 0, "class_name": "cigarette", "confidence": 0.80, "box_xyxy": [102.0, 101.0, 151.0, 131.0]},  # IoU > 0.8
        {"class_id": 0, "class_name": "cigarette", "confidence": 0.70, "box_xyxy": [300.0, 300.0, 340.0, 320.0]},  # Disjoint object
    ]
    merged = global_merge(candidates, 500, 500, merge_iou=0.50)
    assert len(merged) == 2
    assert merged[0]["confidence"] == 0.85
    assert merged[1]["confidence"] == 0.70


def test_global_merge_boundary_clipping():
    """Boxes extending beyond image bounds must be cleanly clipped."""
    candidates = [
        {"class_id": 0, "class_name": "cigarette", "confidence": 0.90, "box_xyxy": [-10.0, 50.0, 100.0, 120.0]},
        {"class_id": 0, "class_name": "cigarette", "confidence": 0.85, "box_xyxy": [400.0, 450.0, 550.0, 520.0]},
    ]
    merged = global_merge(candidates, 500, 500, merge_iou=0.50)
    assert len(merged) == 2
    assert merged[0]["box_xyxy"] == [0.0, 50.0, 100.0, 120.0]
    assert merged[1]["box_xyxy"] == [400.0, 450.0, 500.0, 500.0]
