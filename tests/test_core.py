"""Regression tests for the colour, safety and export fixes.

Run from the project folder:  py -m pytest tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import engine as E  # noqa: E402
from core import exporter as X  # noqa: E402


def _to_cv_lab(L, a, b):
    return np.array([L * 255.0 / 100.0, a + 128.0, b + 128.0])


def test_ciede2000_matches_reference_pair():
    # First pair of Sharma, Wu & Dalal's CIEDE2000 test data: ΔE00 = 2.0425.
    value = E._lab_distance_ciede2000(_to_cv_lab(50, 2.6772, -79.7751), _to_cv_lab(50, 0, -82.7485))
    assert float(value) == pytest.approx(2.0425, abs=1e-3)


def test_white_and_cream_are_different_colors():
    def lab(rgb):
        return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)

    assert float(E._lab_distance_ciede2000(lab((250, 250, 250)), lab((255, 240, 200)))) > 9.0


def test_corner_touch_is_a_loose_piece():
    sheet = np.zeros((50, 50), np.uint8)
    sheet[0:25, 0:25] = 1
    sheet[25:40, 25:40] = 1  # joined to the frame piece only at one corner
    assert len(E.floating_islands(sheet)) == 1


def test_thin_bridge_is_reported():
    sheet = np.zeros((80, 80), np.uint8)
    sheet[:, :20] = 1
    sheet[30:50, 50:70] = 1
    sheet[40, 20:50] = 1  # one-pixel bridge
    assert E.floating_islands(sheet) == []
    kernel = E._bridge_kernel(4.0)  # 0.8 mm at 4 px/mm -> 3 px
    assert len(E.weak_bridges(sheet, kernel)) == 1
    sheet[38:43, 20:50] = 1  # widen the bridge to 5 px
    assert E.weak_bridges(sheet, kernel) == []


@pytest.fixture(scope="module")
def shapes_image(tmp_path_factory):
    rng = np.random.default_rng(3)
    img = np.full((450, 320, 3), 245, np.uint8)
    colors = [(20, 20, 20), (200, 40, 40), (40, 90, 200), (240, 200, 60)]
    for i in range(30):
        c = colors[i % 4]
        x, y = int(rng.integers(20, 300)), int(rng.integers(20, 430))
        if i % 2:
            cv2.circle(img, (x, y), int(rng.integers(6, 40)), c, -1)
        else:
            cv2.ellipse(img, (x, y), (int(rng.integers(8, 50)), int(rng.integers(4, 20))), int(rng.integers(0, 180)), 0, 360, c, -1)
    path = tmp_path_factory.mktemp("img") / "shapes.png"
    cv2.imwrite(str(path), img[:, :, ::-1])
    return path


def test_fast_compaction_equals_full_search(shapes_image):
    project = E.analyze_image(shapes_image, 210, 297, 5, max_dimension=600)
    layers = E._initial_layers(project["components"], project["component_map"], project["palette"])
    fast = E.compact_layers(layers, project["component_map"])
    full = E._compact_layers_full(layers, project["component_map"])
    key = lambda stack: [(l["color_index"], sorted(l["component_ids"])) for l in stack]  # noqa: E731
    assert key(fast) == key(full)


def test_analysis_is_safe_and_every_component_used_once(shapes_image):
    project = E.analyze_image(shapes_image, 210, 297, 5, max_dimension=600)
    assert project["validation"]["safe"]
    owned = [cid for layer in project["layers"] for cid in layer["component_ids"]]
    assert sorted(owned) == list(range(len(project["components"])))
    for sheet in project["sheets"]:
        assert cv2.connectedComponents(sheet, connectivity=4)[0] - 1 == 1


def test_edge_fringe_is_absorbed_but_thin_line_kept(tmp_path):
    img = np.full((600, 420, 3), 255, np.uint8)
    cv2.circle(img, (210, 300), 140, (20, 20, 20), -1)
    cv2.circle(img, (210, 300), 110, (240, 200, 60), -1)
    for i in range(60):
        img[100 + i, 40 + i] = (20, 20, 20)  # deliberate thin diagonal line
    path = tmp_path / "rings.png"
    cv2.imwrite(str(path), img[:, :, ::-1])
    project = E.analyze_image(path, 210, 297, 6)
    for layer in project["layers"]:
        # No paper sheet may consist of scattered edge fragments.
        assert len(layer["component_ids"]) <= 2


def test_export_checks_traced_outline_and_keeps_existing_folder(shapes_image, tmp_path):
    project = E.analyze_image(shapes_image, 210, 297, 4, max_dimension=500)
    for sheet in project["sheets"]:
        assert X.check_traced_sheet(sheet, E.project_px_per_mm(project))["floating"] == 0
    existing = tmp_path / "Demo_Export"
    existing.mkdir()
    (existing / "keep.txt").write_text("user file")
    root = X.export_package(project, tmp_path, "Demo")
    assert root.name == "Demo_Export_2"
    assert (existing / "keep.txt").read_text() == "user file"
    assert len(list((root / "SVG").glob("*.svg"))) == len(project["layers"])
    reloaded, name = X.load_project(next((root / "Project").glob("*.colorbox")))
    assert name == "Demo" and reloaded["validation"]["safe"]
