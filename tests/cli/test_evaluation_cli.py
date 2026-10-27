import json

import pytest


def records():
    return {
        "images": [{"image_id": "synthetic", "width": 640, "height": 640,
                    "gt": [[0, [0, 0, 20, 20]], [0, [100, 100, 150, 150]]]}],
        "predictions": [{"image_id": "synthetic", "class_id": 0, "score": .9,
                         "box": [0, 0, 14, 20]}],
    }


def test_score_uses_corrected_ap75_and_fixed_cohorts():
    from p05_detect.eval.__main__ import score_records

    result = score_records(records())
    assert result["all"]["AP50"] == pytest.approx(51 / 101)
    assert result["all"]["AP75"] == 0
    assert 0 < result["all"]["AP50:95"] < result["all"]["AP50"]
    assert result["small"]["n_gt"] == 1
    assert result["nonsmall"]["n_gt"] == 1
    assert result["all"]["precision"] == 1
    assert result["all"]["recall_50_c25"] == .5
    assert result["all"]["fp_per_image"] == 0


@pytest.mark.parametrize("change", [
    lambda r: r["images"].append(r["images"][0]),
    lambda r: r["images"][0].update(width=0),
    lambda r: r["images"][0].update(gt=[[0, [2, 2, 1, 1]]]),
    lambda r: r["predictions"][0].update(image_id="unknown"),
    lambda r: r["predictions"][0].update(score=float("nan")),
    lambda r: r["images"].clear(),
])
def test_untrusted_records_fail_closed(change):
    from p05_detect.eval.__main__ import score_records

    data = records()
    change(data)
    with pytest.raises(ValueError):
        score_records(data)


def test_cli_scores_without_private_data(tmp_path, capsys):
    from p05_detect.eval.__main__ import main

    path = tmp_path / "records.json"
    path.write_text(json.dumps(records()))
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["small"]["AP75"] == 0
    path.write_text("not json")
    assert main([str(path)]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_rejects_numeric_overflow(tmp_path, capsys):
    from p05_detect.eval.__main__ import main

    data = records()
    data["predictions"][0]["box"][2] = 10 ** 400
    path = tmp_path / "overflow.json"
    path.write_text(json.dumps(data))
    assert main([str(path)]) == 2
    assert "error:" in capsys.readouterr().err
