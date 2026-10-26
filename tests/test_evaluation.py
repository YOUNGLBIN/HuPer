import pytest

from huper.evaluate import (
    METRIC_NAMES,
    aggregate_subject_metrics,
    validate_caption_keys,
)


def test_caption_keys_require_complete_reference_set():
    references = {"0": ["zero"], "1": ["one"]}
    with pytest.raises(KeyError, match="missing"):
        validate_caption_keys(references, {"0": "zero"}, require_complete=True)
    assert validate_caption_keys(references, {"0": "zero"}, require_complete=False) == ["0"]


def test_subject_metric_average():
    subject_metrics = {
        1: {metric: 0.2 for metric in METRIC_NAMES},
        2: {metric: 0.4 for metric in METRIC_NAMES},
        5: {metric: 0.6 for metric in METRIC_NAMES},
        7: {metric: 0.8 for metric in METRIC_NAMES},
    }
    average = aggregate_subject_metrics(subject_metrics)
    assert set(average) == set(METRIC_NAMES)
    assert all(value == pytest.approx(0.5) for value in average.values())
