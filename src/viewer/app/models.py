"""
Data models structured to be identical to the format specified by asxtrade.py
and the free ASX API endpoint
"""
import django.db.models as model
from django.conf import settings
from django.forms.models import model_to_dict
from djongo.models import ObjectIdField, DjongoManager
from djongo.models.json import JSONField
from app.messages import warning
from collections import defaultdict
from datetime import datetime, timedelta, date
import re
import io
from functools import lru_cache
import pandas as pd


def validate_stock(stock):
    assert stock is not None
    assert isinstance(stock, str) and len(stock) >= 3
    assert re.match(r"^\w+$", stock)


def validate_date(d):
    assert isinstance(d, str) and len(d) < 20  # YYYY-mm-dd must be less than 20
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", d)


def validate_user(user):
    assert user is not None
    assert user.is_active
    assert user.is_authenticated
    assert not user.is_anonymous
    return user  # fluent style convenience


class Quotation(model.Model):
    _id = ObjectIdField()
    error_code = model.TextField(max_length=100)  # ignore record iff set to non-empty
    error_descr = model.TextField(max_length=100)
    fetch_date = model.TextField(
        null=False, blank=False
    )  # used as an index (string) always YYYY-mm-dd
    asx_code = model.TextField(blank=False, null=False, max_length=20)
    annual_dividend_yield = model.FloatField()
    average_daily_volume = model.IntegerField()
    bid_price = model.FloatField()
    change_in_percent = model.FloatField()
    change_price = model.FloatField()
    code = model.TextField(blank=False, null=False, max_length=20)
    day_high_price = model.FloatField()
    day_low_price = model.FloatField()
    deprecated_market_cap = (
        model.IntegerField()
    )  # NB: deprecated use market_cap instead
    deprecated_number_of_shares = model.IntegerField()
    descr_full = model.TextField(max_length=100)  # eg. Ordinary Fully Paid
    eps = model.FloatField()
    isin_code = model.TextField(max_length=100)
    last_price = model.FloatField()
    last_trade_date = model.DateField()
    market_cap = model.IntegerField()
    number_of_shares = model.IntegerField()
    offer_price = model.FloatField()
    open_price = model.FloatField()
    pe = model.FloatField()
    previous_close_price = model.FloatField()
    previous_day_percentage_change = model.FloatField()
    suspected = model.BooleanField()
    volume = model.IntegerField()
    year_high_date = model.DateField()
    year_high_price = model.FloatField()
    year_low_date = model.DateField()
    year_low_price = model.FloatField()

    objects = DjongoManager()  # convenient access to mongo API

    def __str__(self):
        assert self is not None
        return str(model_to_dict(self))

    def is_error(self):
        if self.error_code is None:
            return False
        return len(self.error_code) > 0

    def eps_as_cents(self):
        if any([self.is_error(), self.eps is None]):
            return 0.0
        return self.eps * 100.0

    def volume_as_millions(self):
        """
        Return the volume as a formatted string (rounded to 2 decimal place)
        represent the millions of dollars transacted for a given quote
        """
        if any([self.is_error(), self.volume is None, self.last_price is None]):
            return ""

        return "{:.2f}".format(self.volume * self.last_price / 1000000.0)

    class Meta:
        db_table = "asx_prices"
        managed = False  # managed by asxtrade.py


class Security(model.Model):
    # eg. { "_id" : ObjectId("5efe83dd4b1fe020d5ba2de8"), "asx_code" : "T3DAC",
    #  "asx_isin_code" : "AU0000T3DAC0", "company_name" : "333D LIMITED",
    # "last_updated" : ISODate("2020-07-26T00:49:11.052Z"),
    # "security_name" : "OPTION EXPIRING 18-AUG-2018 RESTRICTED" }
    _id = ObjectIdField()
    asx_code = model.TextField(blank=False, null=False)
    asx_isin_code = model.TextField()
    company_name = model.TextField()
    last_updated = model.DateField()
    security_name = model.TextField()

    objects = DjongoManager()

    class Meta:
        db_table = "asx_isin"
        managed = False  # managed by asxtrade.py


class CompanyDetails(model.Model):
    # { "_id" : ObjectId("5eff01d14b1fe020d5453e8f"), "asx_code" : "NIC", "delisting_date" : null,
    # "fax_number" : "02 9221 6333", "fiscal_year_end" : "31/12", "foreign_exempt" : false,
    # "industry_group_name" : "Materials",
    # "latest_annual_reports" : [ { "id" : "02229616", "document_release_date" : "2020-04-29T14:45:12+1000",
    # "document_date" : "2020-04-29T14:39:36+1000", "url" : "http://www.asx.com.au/asxpdf/20200429/pdf/44hc5731pmh9mw.pdf", "relative_url" : "/asxpdf/20200429/pdf/44hc5731pmh9mw.pdf", "header" : "Annual Report and Notice of AGM", "market_sensitive" : false, "number_of_pages" : 118, "size" : "4.0MB", "legacy_announcement" : false }, { "id" : "02209126", "document_release_date" : "2020-02-28T18:09:26+1100", "document_date" : "2020-02-28T18:06:25+1100", "url" : "http://www.asx.com.au/asxpdf/20200228/pdf/44fm8tp5qy0k7x.pdf", "relative_url" : "/asxpdf/20200228/pdf/44fm8tp5qy0k7x.pdf",
    # "header" : "Annual Report and Appendix 4E", "market_sensitive" : true, "number_of_pages" : 64,
    # "size" : "1.6MB", "legacy_announcement" : false }, { "id" : "02163933", "document_release_date" :
    # "2019-10-25T11:50:50+1100", "document_date" : "2019-10-25T11:48:43+1100",
    # "url" : "http://www.asx.com.au/asxpdf/20191025/pdf/449w6d0phvgr05.pdf",
    # "relative_url" : "/asxpdf/20191025/pdf/449w6d0phvgr05.pdf", "header" : "Annual Report and Notice of AGM",
    # "market_sensitive" : false, "number_of_pages" : 74, "size" : "2.5MB", "legacy_announcement" : false } ],
    # "listing_date" : "2018-08-20T00:00:00+1000",
    # "mailing_address" : "Level 2, 66 Hunter Street, SYDNEY, NSW, AUSTRALIA, 2000",
    # "name_abbrev" : "NICKELMINESLIMITED", "name_full" : "NICKEL MINES LIMITED",
    # "name_short" : "NICKLMINES", "phone_number" : "02 9300 3311",
    # "primary_share" : { "code" : "NIC", "isin_code" : "AU0000018236", "desc_full" : "Ordinary Fully Paid",
    # "last_price" : 0.61, "open_price" : 0.595, "day_high_price" : 0.615, "day_low_price" : 0.585,
    # "change_price" : 0.02, "change_in_percent" : "3.39%", "volume" : 3127893, "bid_price" : 0.605,
    # "offer_price" : 0.61, "previous_close_price" : 0.59, "previous_day_percentage_change" : "1.724%",
    # "year_high_price" : 0.731, "last_trade_date" : "2020-07-24T00:00:00+1000",
    # "year_high_date" : "2019-09-24T00:00:00+1000", "year_low_price" : 0.293,
    # "year_low_date" : "2020-03-18T00:00:00+1100", "pe" : 8.73, "eps" : 0.0699,
    # "average_daily_volume" : 7504062, "annual_dividend_yield" : 0, "market_cap" : 1255578789,
    # "number_of_shares" : 2128099642, "deprecated_market_cap" : 1127131000,
    # "deprecated_number_of_shares" : 1847755410, "suspended" : false,
    # "indices" : [ { "index_code" : "XKO", "name_full" : "S&P/ASX 300", "name_short" : "S&P/ASX300",
    # "name_abrev" : "S&P/ASX 300" }, { "index_code" : "XAO", "name_full" : "ALL ORDINARIES",
    # "name_short" : "ALL ORDS", "name_abrev" : "All Ordinaries" }, { "index_code" : "XSO",
    # "name_full" : "S&P/ASX SMALL ORDINARIES", "name_short" : "Small Ords",
    # "name_abrev" : "S&P/ASX Small Ords" }, { "index_code" : "XMM",
    # "name_full" : "S&P/ASX 300 Metals and Mining (Industry)", "name_short" : "MTL&MINING",
    # "name_abrev" : "Metals and Mining" } ] }, "primary_share_code" : "NIC",
    # "principal_activities" : "Nickel pig iron and nickel ore production.",
    # "products" : [ "shares" ], "recent_announcement" : false,
    # "registry_address" : "Level 3, 60 Carrington Street, SYDNEY, NSW, AUSTRALIA, 2000",
    # "registry_name" : "COMPUTERSHARE INVESTOR SERVICES PTY LIMITED",
    # "registry_phone_number" : "02 8234 5000", "sector_name" : "Metals & Mining",
    # "web_address" : "http://www.nickelmines.com.au/" }
    _id = ObjectIdField()
    asx_code = model.TextField(null=False, blank=False)
    delisting_date = model.TextField(null=True)  # if null, not delisted
    name_full = model.TextField(null=False, blank=False)
    phone_number = model.TextField(null=False, blank=True)
    bid_price = model.FloatField()
    offer_price = model.FloatField()
    latest_annual_reports = JSONField()
    previous_close_price = model.FloatField()
    average_daily_volume = model.IntegerField()
    number_of_shares = model.IntegerField()
    suspended = model.BooleanField()
    indices = JSONField()
    primary_share_code = model.TextField()
    principal_activities = model.TextField()
    products = JSONField()
    recent_announcement = model.BooleanField()
    registry_address = model.TextField()
    registry_name = model.TextField()
    registry_phone_number = model.TextField()
    sector_name = model.TextField()
    web_address = model.TextField()

    objects = DjongoManager()

    class Meta:
        managed = False
        db_table = "asx_company_details"


class Watchlist(model.Model):
    id = ObjectIdField(unique=True, db_column="_id")
    # record stocks of interest to the user
    user = model.ForeignKey(settings.AUTH_USER_MODEL, on_delete=model.CASCADE)
    asx_code = model.TextField()
    when = model.DateTimeField(auto_now_add=True)

    objects = DjongoManager()

    class Meta:
        managed = True  # viewer application is responsible NOT asxtrade.py
        db_table = "user_watchlist"


def user_watchlist(user):
    hits = Watchlist.objects.filter(user=user).values_list("asx_code", flat=True)
    results = set(hits)
    print("Found {} stocks in user watchlist".format(len(results)))
    return results

@lru_cache(maxsize=16)
def all_available_dates(reference_stock="ANZ"):
    """
    Returns a sorted list of available dates where the reference stock has a price. stocks
    which are suspended/delisted may have limited dates. The list is sorted from oldest to 
    newest (ascending sort). As this is a frequently used query, an LRU cache is implemented 
    to avoid hitting the database too much.
    """
    # use reference_stock to quickly search the db by limiting the stocks searched
    dates = Quotation.objects.mongo_distinct(
        "fetch_date", {"asx_code": reference_stock}
    )
    ret = sorted(dates, key=lambda k: datetime.strptime(k, "%Y-%m-%d"))
    return ret


def stocks_by_sector():
    rows = [
        d
        for d in CompanyDetails.objects.values("asx_code", "sector_name").order_by(
            "asx_code"
        )
    ]
    df = pd.DataFrame.from_records(rows)
    assert len(df) > 0
    assert "asx_code" in df.columns and "sector_name" in df.columns
    return df


class Sector(model.Model):
    """
    Table of ASX sector (GICS) names. Manually curated for now.
    """
    id = ObjectIdField(unique=True, db_column="_id")
    sector_name = model.TextField(unique=True)
    sector_id = model.IntegerField(db_column="id")

    objects = DjongoManager()

    class Meta:
        managed = False
        db_table = "sector"

def all_sectors():
    iterable = list(CompanyDetails.objects.order_by().values_list('sector_name', flat=True).distinct())
    #print(iterable)
    results = [
        (sector, sector) for sector in iterable
    ]  # as tuples since we want to use it in django form choice field
    return results


def all_sector_stocks(sector_name):
    """
    Return a set of unique ASX stock codes for every security designated as part of the specified sector
    """
    assert sector_name is not None and len(sector_name) > 0
    stocks = set(
        CompanyDetails.objects.order_by("asx_code")
        .filter(sector_name=sector_name)
        .values_list("asx_code", flat=True)
    )
    return stocks


def desired_dates(
    today=None, start_date=None
):  # today is provided as keyword arg for testing
    """
    Return a list of contiguous dates from [today-n_days thru to today inclusive] as 'YYYY-mm-dd' strings, Ordered
    from start_date thru today inclusive. Start_date may be:
    1) a string in YYYY-mm-dd format OR
    2) a datetime instance OR
    3) a integer number of days (>0) from today backwards to return.
    """
    if today is None:
        today = date.today()
    if isinstance(start_date, (datetime, date)):
        pass  # FALLTHRU
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    elif isinstance(start_date, int):
        assert start_date > 0
        start_date = today - timedelta(days=start_date - 1)  # -1 for today inclusive
    else:
        raise ValueError("Unsupported start_date {}".format(type(start_date)))
    assert start_date <= today
    all_dates = [
        d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, today, freq="D")
    ]
    assert len(all_dates) > 0
    return sorted(all_dates, key=lambda d: datetime.strptime(d, "%Y-%m-%d"))


def all_stocks():
    all_securities = Security.objects.values_list("asx_code", flat=True)
    return set(all_securities)

def find_movers(threshold, required_dates):
    """
    Return a dataframe with row index set to ASX ticker symbols and the only column set to 
    the sum over all desired dates for percentage change in the stock price. A negative sum
    implies a decrease, positive an increase in price over the observation period.
    """
    assert threshold >= 0.0
    assert desired_dates is not None
    cip = company_prices(all_stocks(), required_dates, fields="change_in_percent")
    movements = cip.sum(axis=1)
    return movements[movements.abs() >= threshold]


def find_named_companies(wanted_name, wanted_activity):
    ret = set()
    if len(wanted_name) > 0:
        # match by company name first...
        ret.update(
            CompanyDetails.objects.filter(name_full__icontains=wanted_name).values_list(
                "asx_code", flat=True
            )
        )
        # but also matching codes
        ret.update(
            CompanyDetails.objects.filter(asx_code__icontains=wanted_name).values_list(
                "asx_code", flat=True
            )
        )
        # and if the code has no details, we try another method to find them...
        # by checking all codes seen on the most recent date
        latest_date = latest_quotation_date("ANZ")
        ret.update(
            Quotation.objects.filter(asx_code__icontains=wanted_name)
            .filter(fetch_date=latest_date)
            .values_list("asx_code", flat=True)
        )

    if len(wanted_activity) > 0:
        ret.update(
            CompanyDetails.objects.filter(
                principal_activities__icontains=wanted_activity
            ).values_list("asx_code", flat=True)
        )
    return ret


def latest_quotation_date(stock):
    d = all_available_dates(reference_stock=stock)
    return d[-1]


def all_quotes(stock, all_dates=None):
    """
    company_prices() is better as it is pre-pivoted monthly tables, but this is needed for some use cases
    """
    assert len(stock) >= 3
    if all_dates is None:
        all_dates = desired_dates(start_date=30)
    quotes = (
        Quotation.objects.filter(asx_code=stock)
        .filter(fetch_date__in=all_dates)
        .exclude(error_code="id-or-code-invalid")
    )
    rows = [model_to_dict(quote) for quote in quotes]
    df = pd.DataFrame.from_records(rows)
    return df


def latest_quote(stocks):
    """
    If stocks is a str, retrieves the latest quote and returns a tuple (Quotation, latest_date).
    If stocks is None, returns a tuple (queryset, latest_date) of all stocks.
    If stocks is an iterable, returns a tuple (queryset, latest_date) of selected stocks
    """
    if isinstance(stocks, str):
        latest_date = latest_quotation_date(stocks)
        obj = Quotation.objects.get(asx_code=stocks, fetch_date=latest_date)
        return (obj, latest_date)
    else:
        latest_date = latest_quotation_date("ANZ")
        qs = Quotation.objects.filter(fetch_date=latest_date)
        if stocks is not None:
            if len(stocks) == 1:
                qs = qs.filter(asx_code=stocks[0])
            else:
                qs = qs.filter(asx_code__in=stocks)
        return (qs, latest_date)


def make_superdf(required_tags, stock_codes):
    assert required_tags is not None and len(required_tags) >= 1
    assert stock_codes is None or len(stock_codes) > 0  # NB: zero stocks considered bad
    dataframes = MarketDataCache.objects.filter(
        tag__in=required_tags, dataframe_format="parquet"
    ).values_list("dataframe", flat=True)
    superdf = None
    n = 0
    for parquet_bytes in dataframes:
        n += 1
        with io.BytesIO(parquet_bytes) as fp:
            df = pd.read_parquet(fp)
            if (
                len(df) == 0
            ):  # skip empty frames: not that persist_dataframes.py has a bug where the matrix has wrong/index columns when empty so be careful not to merge them!
                continue
            # remove rows which are not relevant before merge to speed things...
            if stock_codes is not None:
                # print("Before {}".format(len(df)))
                df = df.reindex(tuple(stock_codes))
                # print("After {} (had {} stocks)".format(len(df), len(stock_codes)))
            # print(df)
            if superdf is None:
                superdf = df
            else:
                superdf = superdf.merge(
                    df, how="outer", left_index=True, right_index=True
                )
    return (superdf, n)


def day_low_high(stock, all_dates=None):
    """
    For the specified dates (specified in strict YYYY-mm-dd format) return
    the day low/high price, last price and volume as a pandas dataframe with dates down and
    columns which represent the prices and volume (in $AUD millions).
    """
    assert stock is not None and len(stock) >= 3

    quotes = (
        Quotation.objects.filter(asx_code=stock)
        .filter(fetch_date__in=all_dates)
        .exclude(error_code="id-or-code-invalid")
    )
    rows = []
    for q in quotes:
        rows.append(
            {
                "date": q.fetch_date,
                "day_low_price": q.day_low_price,
                "day_high_price": q.day_high_price,
                "volume": q.volume_as_millions(),
                "last_price": q.last_price,
            }
        )
    day_low_high_df = pd.DataFrame.from_records(rows)
    day_low_high_df.set_index(day_low_high_df["date"], inplace=True)
    return day_low_high_df


def impute_missing(df, method="linear"):
    assert df is not None
    if method == "linear":  # faster...
        result = df.interpolate(
            method=method, limit_direction="forward", axis="columns"
        )
        return result
    else:
        # must have a DateTimeIndex so...
        df.columns = pd.to_datetime(df.columns)
        df = df.interpolate(method=method, limit_direction="forward", axis="columns")
        df.set_index(
            df.index.format(), inplace=True
        )  # convert back to strings for caller compatibility
        return df


def all_etfs():
    etf_codes = [
        s.asx_code
        for s in Security.objects.filter(
            security_name="EXCHANGE TRADED FUND UNITS FULLY PAID"
        )
    ]
    print("Found {} ETF codes".format(len(etf_codes)))
    return etf_codes


def increasing_eps(stock_codes, past_n_days=300):
    all_dates = desired_dates(start_date=past_n_days)
    required_tags = set(
        ["eps-{}-{}-asx".format(date[5:7], date[0:4]) for date in all_dates]
    )
    # NB: we dont care here if some tags cant be found
    df, n = make_superdf(required_tags, stock_codes)
    # df will be very large: 300 days * ~2000 stocks... but mostly the numbers will be the same each day...
    # at least 2c per share positive max(eps) is required to be considered significant
    increasing_eps_stocks = [
        idx
        for idx, series in df.iterrows()
        if series.is_monotonic_increasing and max(series) >= 0.02
    ]
    return increasing_eps_stocks


def increasing_yield(stock_codes, past_n_days=300):
    all_dates = desired_dates(start_date=past_n_days)
    required_tags = set(
        [
            "annual_dividend_yield-{}-{}-asx".format(date[5:7], date[0:4])
            for date in all_dates
        ]
    )
    df, n = make_superdf(required_tags, stock_codes)
    # ignore penny-ante stocks (must be at least 1c per share dividend)
    increasing_yield_stocks = [
        idx
        for idx, series in df.iterrows()
        if series.is_monotonic_increasing and max(series) >= 0.01
    ]
    return increasing_yield_stocks

def get_required_tags(all_dates, fields):
    required_tags = set()
    for d in all_dates:
        validate_date(d)
        yyyy = d[0:4]
        mm = d[5:7]
        required_tags.add("{}-{}-{}-asx".format(fields, mm, yyyy))
    return required_tags

def company_prices(
        stock_codes,
        all_dates=None,
        fields="last_price",
        fail_missing_months=True,
        missing_cb=impute_missing, # or None if you want missing values 
):
    """
    Return a dataframe with the required companies (iff quoted) over the
    specified dates. By default last_price is provided. Fields may be a list,
    in which case the dataframe has columns for each field and dates are rows (in this case only one stock is permitted)
    """
    if not isinstance(fields, str):  # assume iterable if not str...
        assert len(stock_codes) == 1
        dataframes = [
            company_prices(
                stock_codes,
                all_dates=all_dates,
                fields=field,
                fail_missing_months=fail_missing_months,
            )
            for field in fields
        ]
        result_df = pd.concat(dataframes, ignore_index=True)
        result_df.set_index(pd.Index(fields), inplace=True)
        # print(result_df)
        result_df = result_df.transpose()
        # print(result_df)
        assert list(result_df.columns) == fields
        # reject rows which are all NA to avoid downstream problems eg. plotting stocks
        # NB: we ONLY do this for the multi-field case, single field it is callers responsibility
        result_df = result_df.dropna()
        return result_df

    # print(stock_codes)
    assert isinstance(fields, str)
    if all_dates is None:
        all_dates = [datetime.strftime(datetime.now(), "%Y-%m-%d")]

    required_tags = get_required_tags(all_dates, fields)
    which_cols = set(all_dates)
    # construct a "super" dataframe from the constituent parquet data
    superdf, n_dataframes = make_superdf(required_tags, stock_codes)

    # drop columns not present in all_dates to ensure we are giving just the results requested
    cols_to_drop = [date for date in superdf.columns if date not in which_cols]
    superdf = superdf.drop(columns=cols_to_drop)

    # on the first of the month, we dont have data yet so we permit one missing tag for this reason
    if fail_missing_months and n_dataframes < len(required_tags) - 1:
        raise ValueError(
            "Not all required data is available - aborting! Found {} wanted {}".format(
                n_dataframes, required_tags
            )
        )
    # NB: ensure all columns are ALWAYS in ascending date order
    dates = sorted(
        list(superdf.columns), key=lambda k: datetime.strptime(k, "%Y-%m-%d")
    )
    superdf = superdf[dates]
    if missing_cb is not None and superdf.isnull().values.any():
        superdf = missing_cb(superdf)
    return superdf


class MarketDataCache(model.Model):
    # { "_id" : ObjectId("5f44c54457d4bb6dfe6b998f"), "scope" : "all-downloaded",
    # "tag" : "change_price-05-2020-asx", "dataframe_format" : "parquet",
    # "field" : "change_price", "last_updated" : ISODate("2020-08-25T08:01:08.804Z"),
    # "market" : "asx", "n_days" : 3, "n_stocks" : 0,
    # "sha256" : "75d0ad7e057621e6a73508a178615bcc436d97110bcc484f1cfb7d478475abc5",
    # "size_in_bytes" : 2939, "status" : "INCOMPLETE" }
    size_in_bytes = model.IntegerField()
    status = model.TextField()
    tag = model.TextField()
    dataframe_format = model.TextField()
    field = model.TextField()
    last_updated = model.DateTimeField()
    market = model.TextField()
    n_days = model.IntegerField()
    n_stocks = model.IntegerField()
    sha256 = model.TextField()
    _id = ObjectIdField()
    scope = model.TextField()
    dataframe = model.BinaryField()

    objects = DjongoManager()
    
    class Meta:
        managed = False  # table is managed by persist_dataframes.py
        db_table = "market_quote_cache"


class VirtualPurchase(model.Model):
    id = ObjectIdField(unique=True, db_column="_id")
    user = model.ForeignKey(settings.AUTH_USER_MODEL, on_delete=model.CASCADE)
    asx_code = model.TextField(max_length=10)
    buy_date = model.DateField()
    price_at_buy_date = model.FloatField()
    amount = model.FloatField()  # dollar value purchased (assumes no fees)
    n = (
        model.IntegerField()
    )  # number of shares purchased at buy_date (rounded down to nearest whole share)

    objects = DjongoManager()

    def current_price(self):
        assert self.n > 0
        q, as_at = latest_quote(self.asx_code)
        p = q.last_price
        buy_price = self.price_at_buy_date
        if buy_price > 0:
            pct_move = (p / buy_price) * 100.0 - 100.0
        else:
            pct_move = 0.0
        return (self.n * p, pct_move)

    def __str__(self):
        cur_price, pct_move = self.current_price()
        return "If purchased on {}: ${} ({} shares) are now worth ${:.2f} ({:.2f}%)".format(
            self.buy_date, self.amount, self.n, cur_price, pct_move
        )

    class Meta:
        managed = True  # viewer application
        db_table = "virtual_purchase"


class ImageCache(model.Model):
    # some images in viewer app are expensive to compute, so we cache them
    # and if less than a week old, use them rather than recompute. The views
    # automatically check the cache for expensive images eg. sector performance plot
    _id = ObjectIdField()
    base64 = model.TextField(max_length=400 * 1024, null=False, blank=False)
    tag = model.TextField(
        max_length=100, null=False, blank=False
    )  # used to identify if image is in ImageCache
    last_updated = model.DateTimeField(auto_now_add=True)
    valid_until = model.DateTimeField(
        auto_now_add=True
    )  # image is cached until is_outdated(), up to caller to set this value correctly

    def is_outdated(self):
        return self.last_updated > self.valid_until

    class Meta:
        managed = True
        db_table = "image_cache"


def update_image_cache(tag, base64_data, valid_days=7):  # cache for a week by default
    assert base64_data is not None
    assert len(tag) > 0
    now = datetime.utcnow()
    defaults = {
        "base64": base64_data,
        "tag": tag,
        "last_updated": now,
        "valid_until": now + timedelta(days=valid_days),
    }
    ImageCache.objects.update_or_create(tag=tag, defaults=defaults)


def user_purchases(user):
    """
    Returns a dict: asx_code -> VirtualPurchase of the specified user's watchlist
    """
    validate_user(user)
    purchases = defaultdict(list)
    for purchase in VirtualPurchase.objects.filter(user=user):
        code = purchase.asx_code
        purchases[code].append(purchase)
    print("Found virtual purchases for {} stocks".format(len(purchases)))
    return purchases
