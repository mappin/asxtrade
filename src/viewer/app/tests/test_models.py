from datetime import datetime
import pytest
from app.models import (
    validate_stock,
    validate_date,
    desired_dates,
    Quotation,
    Watchlist,
    all_sectors,
    all_sector_stocks,
    all_stocks,
    user_watchlist,
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
    u1 = django_user_model.objects.create(username='U1', password='U1')
    u2 = django_user_model.objects.create(username='u2', password='u2')
    Watchlist.objects.create(user=u1, asx_code='ASX1')
    assert u2.is_active and u1.is_active

@pytest.mark.django_db
def test_user_watchlist(uw_fixture, django_user_model):
    u1 = django_user_model.objects.get(username='U1')
    assert user_watchlist(u1) == set(['ASX1'])
    u2 = django_user_model.objects.get(username='u2')
    assert user_watchlist(u2) == set()
