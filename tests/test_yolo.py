from pitchview.yolo import YOLO_CLASSES, yolo_lines


def test_yolo_lines_are_relative_centre_boxes_with_classes_from_zero():
    coco = {
        "images": [{"id": 7, "file_name": "a.jpg", "width": 200, "height": 100}, {"id": 8, "file_name": "b.jpg",
                   "width": 200, "height": 100}],
        "annotations": [
            {"image_id": 7, "category_id": 1, "bbox": [10, 20, 40, 60]},  # player
            {"image_id": 7, "category_id": 4, "bbox": [100, 50, 10, 10]},  # ball
            {"image_id": 7, "category_id": 0, "bbox": [0, 0, 5, 5]},  # Roboflow's parent category: skipped
        ],
    }  # fmt: skip
    lines = yolo_lines(coco)
    assert lines["a.jpg"] == ["0 0.150000 0.500000 0.200000 0.600000", "3 0.525000 0.550000 0.050000 0.100000"]
    assert lines["b.jpg"] == []  # frames with nobody still get an (empty) label file
    assert YOLO_CLASSES == ["player", "goalkeeper", "referee", "ball"]
