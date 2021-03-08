import pytest
from app.models import validate_stock, validate_date, desired_dates, Quotation, Security, CompanyDetails, all_sectors
from datetime import datetime
from app.factories import CompanyDetailsFactory

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
    with pytest.raises(ValueError):
        desired_dates(0)

    now = datetime.fromisoformat("2020-08-23 07:44:41.398773")
    ret = desired_dates(start_date=7, today=now)
    assert ret == ['2020-08-17', '2020-08-18', '2020-08-19',
                   '2020-08-20', '2020-08-21', '2020-08-22', '2020-08-23']

def test_quotation_error_handling():
    q = Quotation(error_code='404', change_in_percent=4.4)
    assert q.is_error()
    assert q.volume_as_millions() == ""  # since is_error() is True

def test_quotation_volume_handling():
    q = Quotation(error_code='', volume=1000 * 1000, last_price=1.0)
    assert q.volume_as_millions() == "1.00"
    q = Quotation(error_code='', volume=10 * 1000 * 1000, last_price=1.0)
    assert q.volume_as_millions() == "10.00"

def test_quotation_eps_handling(): 
    q = Quotation(error_code='', eps=3.36)
    assert q.eps - 3.36 < 0.001
    assert q.eps_as_cents() == 336.0
    q = Quotation() # eps is None
    assert q.eps is None
    assert q.eps_as_cents() == 0.0

def test_validate_stock():
    bad_stocks = [None, 'AB', '---']
    for stock in bad_stocks:
        with pytest.raises(AssertionError):
            validate_stock(stock)
    good_stocks = ['ABC', 'ABCDE', 'AB2', 'abcde']
    for stock in good_stocks:
        validate_stock(stock)

def test_validate_date():
    bad_dates = ['2020-02-', '', None, '20-02-02', '2020-02-1']
    for d in bad_dates:
        with pytest.raises(AssertionError):
            validate_date(d)
    
    good_dates = ['2020-02-02', '2020-02-01', '2021-12-31']
    for d in good_dates:
        validate_date(d)

@pytest.mark.django_db
def test_all_sectors(company_details):
    assert company_details is not None
    # since company_details_factory gives a single ANZ company details record, this test will work...
    assert all_sectors() == [('Financials', 'Financials')]
