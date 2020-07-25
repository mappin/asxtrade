import django.db.models as model
from djongo.models import ObjectIdField, DjongoManager
from djongo.models.json import JSONField

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

    class Meta:
       db_table = 'asx_prices'
       managed = False # managed by asxtrade.py
