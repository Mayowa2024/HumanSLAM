import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "convert_mapillary_to_yolo_seg.py"
SPEC = importlib.util.spec_from_file_location("mapillary_converter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_taxonomy_has_unique_ids():
    assert len(MODULE.CLASS_NAMES) == 21
    assert len(MODULE.CLASS_TO_ID) == len(MODULE.CLASS_NAMES)
    assert sorted(MODULE.CLASS_TO_ID.values()) == list(range(21))


def test_required_mapillary_classes_are_mapped():
    assert MODULE.MAPILLARY_TO_CLASS["construction--structure--building"] == "building"
    assert MODULE.MAPILLARY_TO_CLASS["construction--barrier--fence"] == "fence"
    assert MODULE.MAPILLARY_TO_CLASS["object--sign--store"] == "store_sign"
    assert MODULE.MAPILLARY_TO_CLASS["object--traffic-sign--front"] == "traffic_sign"


def test_clean_polygon_normalises_and_clips_coordinates():
    polygon = MODULE.clean_polygon(
        [[-10, 0], [100, 0], [100, 50], [0, 50]], 100, 50
    )
    assert polygon.shape == (4, 2)
    assert np.all(polygon >= 0.0)
    assert np.all(polygon <= 1.0)
    assert MODULE.polygon_area(polygon) > 0.9


def test_degenerate_polygon_is_rejected():
    assert MODULE.clean_polygon([[0, 0], [1, 1], [2, 2]], 100, 100) is None
