from copy import deepcopy

import pytest

from nlbse24.constants import LABELS
from nlbse24.evaluation.metrics import evaluate_predictions
from scripts import p1_loo_analysis as analysis


def fixtures():
    result = {}
    for model in analysis.MODELS:
        result[model] = []
        for repo in analysis.REPOSITORIES:
            truth = list(LABELS) * 3
            pred = truth if model == analysis.MODELS[0] else [LABELS[0]] * 9
            result[model].append({
                "protocol": "loo", "seed": 42, "fold": repo,
                "repository": repo.replace("__", "/"), "classes": list(LABELS),
                "data": {"train_fingerprint": "train", "test_fingerprint": repo},
                "model": {"config": {"experiment": {"text_fields":
                    ["title", "body"] if model == analysis.MODELS[0] else ["title"]}}},
                "predictions": {"y_true": truth, "y_pred": pred},
                "metrics": evaluate_predictions(truth, pred),
            })
    return result


def test_p1_analysis_is_paired_and_reproducible(monkeypatch, tmp_path):
    runs = fixtures()
    monkeypatch.setattr(analysis, "load_artifacts", lambda path: runs[path.name])
    first = analysis.analyze(tmp_path, resamples=50)
    assert first == analysis.analyze(tmp_path, resamples=50)
    assert first["title_body_minus_title_only"]["point_difference"] == pytest.approx(5 / 6)
    for a, b in zip(*runs.values(), strict=True):
        b["predictions"] = deepcopy(a["predictions"])
        b["metrics"] = deepcopy(a["metrics"])
    identical = analysis.analyze(tmp_path, resamples=50)
    assert identical["title_body_minus_title_only"]["ci95"] == [0.0, 0.0]


def test_p1_analysis_rejects_unaligned_inputs(monkeypatch, tmp_path):
    runs = fixtures()
    monkeypatch.setattr(analysis, "load_artifacts", lambda path: runs[path.name])
    runs[analysis.MODELS[1]][0]["data"]["test_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="unaligned"):
        analysis.analyze(tmp_path, resamples=50)
    with pytest.raises(ValueError, match="positive"):
        analysis.analyze(tmp_path, resamples=0)
