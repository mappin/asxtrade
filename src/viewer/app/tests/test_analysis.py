import pytest
from app.analysis import (
    as_css_class,
    calculate_trends,
)

def test_as_css_class():
    assert as_css_class(10, -10) == "recent-upward-trend"
    assert as_css_class(-10, 10) == "recent-downward-trend"
    assert as_css_class(0, 0) == "none"