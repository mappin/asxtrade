import pytest
from django.core.exceptions import ValidationError
from app.forms import (
    is_not_blank,
    is_valid_sector,
    SectorSearchForm,
    MoverSearchForm,
)


def test_is_not_blank():
    with pytest.raises(ValidationError):
        is_not_blank("")
    is_not_blank("hello")
    is_not_blank("0")
    is_not_blank("false")


@pytest.mark.django_db
def test_is_valid_sector():
    for item1, item2 in SectorSearchForm.SECTOR_CHOICES:
        # the database is not populated so we cant check return value, but we can check for an exception
        is_valid_sector(item1)


@pytest.mark.django_db
def test_sector_search_form():
    f = SectorSearchForm(data={"sector": SectorSearchForm.SECTOR_CHOICES[0][1]})
    assert f.is_valid()
    f = SectorSearchForm(data={})
    assert not f.is_valid()

def test_mover_search_form():
    f = MoverSearchForm(data={})
    assert not f.is_valid()
    f = MoverSearchForm(data={'threshold': 50.0, 'timeframe_in_days': 10})
    assert f.is_valid()