import django.db.models as model
from django.conf import settings
from djongo.models import ObjectIdField, DjongoManager
from djongo.models.json import JSONField
from pylru import lrudecorator
from datetime import datetime

class Quotation(model.Model):
    _id = ObjectIdField()
    error_code = model.TextField(max_length=100) # ignore record iff set to non-empty
    error_descr = model.TextField(max_length=100)
    fetch_date = model.TextField(null=False, blank=False) # used as an index (string) always YYYY-mm-dd
    asx_code = model.TextField(blank=False, null=False, max_length=20)
    annual_dividend_yield = model.FloatField()
    average_daily_volume = model.IntegerField()
    bid_price = model.FloatField()
    change_in_percent = model.TextField()
    change_price = model.FloatField()
    code = model.TextField(blank=False, null=False, max_length=20)
    day_high_price = model.FloatField()
    day_low_price = model.FloatField()
    deprecated_market_cap = model.IntegerField() # NB: deprecated use market_cap instead
    deprecated_number_of_shares = model.IntegerField()
    descr_full = model.TextField(max_length=100) # eg. Ordinary Fully Paid
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
    previous_day_percentage_change = model.TextField(max_length=100)
    suspected = model.BooleanField()
    volume = model.IntegerField()
    year_high_date = model.DateField()
    year_high_price = model.FloatField()
    year_low_date = model.DateField()
    year_low_price = model.FloatField()

    objects = DjongoManager() # convenient access to mongo API

    def percent_change(self):
        # since the ASX uses a string field, we auto-convert to float on the fly
        pc = float(self.change_in_percent.strip('%'))
        return pc

    class Meta:
       db_table = 'asx_prices'
       managed = False # managed by asxtrade.py


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

    class Meta:
        db_table = "asx_isin"
        managed = False # managed by asxtrade.py

class CompanyDetails(model.Model):
    #{ "_id" : ObjectId("5eff01d14b1fe020d5453e8f"), "asx_code" : "NIC", "delisting_date" : null,
    # "fax_number" : "02 9221 6333", "fiscal_year_end" : "31/12", "foreign_exempt" : false,
    #"industry_group_name" : "Materials",
    #"latest_annual_reports" : [ { "id" : "02229616", "document_release_date" : "2020-04-29T14:45:12+1000",
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
    delisting_date = model.TextField(null=True) # if null, not delisted
    name_full = model.TextField(null=False, blank=False)
    phone_number = model.TextField(null=False, blank=True)
    bid_price = model.FloatField()
    offer_price = model.FloatField()
    latest_annual_reports = JSONField()
    previous_close_price = model.FloatField()
    average_daily_volume = model.IntegerField()
    number_of_shared = model.IntegerField()
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
    # record stocks of interest to the user
    user = model.ForeignKey(settings.AUTH_USER_MODEL, on_delete=model.CASCADE)
    asx_code = model.TextField()
    when = model.DateTimeField(auto_now_add=True)

    objects = DjongoManager()

    class Meta:
        managed = True # viewer application is responsible NOT asxtrade.py
        db_table = "user_watchlist"

def user_watchlist(user):
    watch_list = set([hit.asx_code for hit in Watchlist.objects.filter(user=user)])
    print("Found {} stocks in user watchlist".format(len(watch_list)))
    return watch_list

def latest_quotation_date():
    all_available_dates = sorted(Quotation.objects.mongo_distinct('fetch_date'),
                                 key=lambda k: datetime.strptime(k, "%Y-%m-%d"))
    return all_available_dates[-1]

def latest(stock):
    latest_date = latest_quotation_date()
    obj = Quotation.objects.get(asx_code=stock, fetch_date=latest_date)
    return (obj.last_price, latest_date)

def latest_price(stock):
    price, as_at = latest(stock)
    return price

class VirtualPurchase(model.Model):
    user = model.ForeignKey(settings.AUTH_USER_MODEL, on_delete=model.CASCADE)
    asx_code = model.TextField()
    buy_date = model.DateField()
    price_at_buy_date = model.FloatField()
    amount = model.FloatField() # dollar value purchased (assumes no fees)
    n = model.IntegerField()    # number of shares purchased at buy_date (rounded down to nearest whole share)

    objects = DjongoManager()

    def current_price(self):
        assert self.n > 0
        p = latest_price(self.asx_code)
        buy_price = self.price_at_buy_date
        if buy_price > 0:
            pct_move = (p / buy_price) * 100.0 - 100.0
        else:
            pct_move = 0.0
        return (self.n * p, pct_move)

    def __str__(self):
        cur_price, pct_move = self.current_price()
        return "If purchased on {}: ${} ({} shares) are now worth ${:.2f} ({:.2f}%)".format(self.buy_date, self.amount, self.n, cur_price, pct_move)

    class Meta:
        managed = True                # viewer application
        db_table = "virtual_purchase"

def user_purchases(user):
    assert user is not None
    purchases = {}
    for purchase in VirtualPurchase.objects.filter(user=user):
        code = purchase.asx_code
        if not code in purchases:
            purchases[code] = []
        purchases[code].append(purchase)
    print("Found virtual purchases for {} stocks".format(len(purchases)))
    return purchases
