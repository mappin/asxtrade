import pytest
from app.models import validate_stock, validate_date, desired_dates

def test_validate_stock():
    validate_stock('ANZ') # NB: must not assert
    with pytest.raises(AssertionError):
         validate_stock('AN')

def test_validate_date():
    validate_date('2020-01-01')
    validate_date('2020-12-30')
    with pytest.raises(AssertionError):
        validate_date('2020-2-2')

def test_desired_dates():
    # 1. must specify at least 1 day
    with pytest.raises(AssertionError):
        desired_dates(0)

    now = datetime.fromisoformat("2020-08-23 07:44:41.398773")
    ret = desired_dates(7, today=now)
    assert ret == ['2020-08-17', '2020-08-18', '2020-08-19',
                   '2020-08-20', '2020-08-21', '2020-08-22', '2020-08-23']
