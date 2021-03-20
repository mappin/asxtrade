from datetime import datetime
import pytest
from django.db.models.query import QuerySet
import pandas as pd
from app.models import (
    validate_stock,
    validate_sector,
    validate_user,
    validate_date,
    all_available_dates,
    desired_dates,
    stock_info,
    find_user,
    Quotation,
    Watchlist,
    CompanyDetails,
    Security,
    is_in_watchlist,
    all_sectors,
    all_sector_stocks,
    stocks_by_sector,
    all_stocks,
    user_watchlist,
    valid_quotes_only
)

def test_desired_dates():
    # 1. must specify at least 1 day
    with pytest.raises(ValueError):
        desired_dates(0)

    now = datetime.fromisoformat("2020-08-23 07:44:41.398773")
    ret = desired_dates(start_date=7, today=now)
    assert ret == [
        "2020-08-17",
        "2020-08-18",
        "2020-08-19",
        "2020-08-20",
        "2020-08-21",
        "2020-08-22",
        "2020-08-23",
    ]


def test_quotation_error_handling(quotation_factory):
    q = quotation_factory.build(error_code="404")
    assert q.is_error()
    assert q.volume_as_millions() == ""  # since is_error() is True


def test_quotation_volume_handling(quotation_factory):
    q = quotation_factory.build(volume=1000 * 1000, last_price=1.0)
    assert q.volume_as_millions() == "1.00"
    q = quotation_factory.build(volume=10 * 1000 * 1000, last_price=1.0)
    assert q.volume_as_millions() == "10.00"


def test_quotation_eps_handling(quotation_factory):
    q = quotation_factory.build(eps=3.36)
    assert q.eps - 3.36 < 0.001
    assert q.eps_as_cents() == 336.0
    q = Quotation()  # eps is None
    assert q.eps is None
    assert q.eps_as_cents() == 0.0

@pytest.fixture
def crafted_quotation_fixture(quotation_factory):
    Quotation.objects.all().delete() # ensure no cruft ruins test associated with fixture
    quotation_factory.create(asx_code=None, error_code="id-or-code-invalid")
    quotation_factory.create(asx_code='ABC', last_price=None)
    quotation_factory.create(asx_code='ANZ', fetch_date='2021-01-01', last_price=10.10, volume=1)


@pytest.mark.django_db
def test_valid_quotes_only(crafted_quotation_fixture):
    result = valid_quotes_only('2021-01-01')
    assert result is not None
    assert len(result) == 1
    assert result[0].asx_code == 'ANZ'
    assert result[0].fetch_date == '2021-01-01'
    assert result[0].last_price == 10.10
    assert result[0].volume == 1

def test_validate_stock():
    bad_stocks = [None, "AB", "---"]
    for stock in bad_stocks:
        with pytest.raises(AssertionError):
            validate_stock(stock)
    good_stocks = ["ABC", "ABCDE", "AB2", "abcde", "ANZ"]
    for stock in good_stocks:
        validate_stock(stock)


def test_validate_date():
    bad_dates = ["2020-02-", "", None, "20-02-02", "2020-02-1"]
    for d in bad_dates:
        with pytest.raises(AssertionError):
            validate_date(d)

    good_dates = ["2020-02-02", "2020-02-01", "2021-12-31"]
    for d in good_dates:
        validate_date(d)


@pytest.mark.django_db
def test_all_sectors(company_details):
    assert company_details is not None
    # since company_details_factory gives a single ANZ company details record, this test will work...
    ret = all_sectors()
    #print(ret)
    assert ret == [('Financials', 'Financials')]
    # and check the reverse is true: financials -> ANZ
    all_sector_stocks.cache_clear()
    assert all_sector_stocks('Financials') == set(['ANZ'])

@pytest.mark.django_db
def test_all_stocks(security):
    assert security is not None
    assert len(security.asx_code) >= 3
    assert all_stocks() == set([security.asx_code])

@pytest.fixture
def uw_fixture(django_user_model):
    u1 = django_user_model.objects.create(username='U1', password='U1', is_active=False)
    u2 = django_user_model.objects.create(username='u2', password='u2')
    Watchlist.objects.create(user=u1, asx_code='ASX1')
    assert u2.is_active and not u1.is_active

@pytest.mark.django_db
def test_user_watchlist(uw_fixture, django_user_model):
    u1 = django_user_model.objects.get(username='U1')
    assert user_watchlist(u1) == set(['ASX1'])
    u2 = django_user_model.objects.get(username='u2')
    assert user_watchlist(u2) == set()

@pytest.mark.django_db
def test_validate_sector(company_details):
    assert company_details is not None # avoid pylint warning
    validate_sector('Financials') # must not raise exception

@pytest.mark.django_db
def test_validate_user(uw_fixture, django_user_model):
    u1 = django_user_model.objects.get(username='U1')
    # since u1 is not active...
    with pytest.raises(AssertionError):
        validate_user(u1)

    u2 = django_user_model.objects.get(username='u2')
    validate_user(u2)

@pytest.mark.django_db
def test_in_watchlist(uw_fixture):
    find_user.cache_clear()
    assert is_in_watchlist('U1', 'ASX1')
    assert not is_in_watchlist('u2', 'ASX1')


@pytest.mark.django_db
def test_all_available_dates(crafted_quotation_fixture):
    assert all_available_dates() == ['2021-01-01']
    assert all_available_dates(reference_stock='ASX1') == []

@pytest.fixture
def comp_deets(company_details_factory, security_factory):
    CompanyDetails.objects.all().delete()
    Security.objects.all().delete()
    company_details_factory.create(asx_code='ANZ')
    security_factory.create(asx_code='ANZ')

@pytest.mark.django_db
def test_stock_info(comp_deets):
    t = stock_info('ANZ')
    assert isinstance(t, tuple)
    assert len(t) == 2
    assert len(t[0]) == 1
    assert isinstance(t[0], QuerySet)
    s = t[0].first()
    assert s.asx_code == 'ANZ'
    assert s.asx_isin_code == 'ISIN000001'
    assert isinstance(t[1], CompanyDetails)
    assert t[1].asx_code == 'ANZ'

@pytest.mark.django_db
def test_stocks_by_sector(comp_deets):
    df = stocks_by_sector()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0].asx_code == 'ANZ'
    assert df.iloc[0].sector_name == 'Financials'
