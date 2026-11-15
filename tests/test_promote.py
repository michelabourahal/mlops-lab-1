import pytest

from src.food11.promote import should_promote, standard_error


def test_standard_error():
    assert standard_error(0.5, 100) == pytest.approx(0.05)


def test_gate_requires_more_than_two_standard_errors():
    # champion at 0.75 on 1096 test images: se ~ 0.0131, so the bar is ~ 0.7762
    assert not should_promote(0.76, 0.75, 1096)
    assert should_promote(0.78, 0.75, 1096)
