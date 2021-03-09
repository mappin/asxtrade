from django.template.defaulttags import register
import re
from functools import lru_cache
from app.models import CompanyDetails

@register.filter
def get_item(d, key):
    assert isinstance(d, dict)
    assert isinstance(key, str)
    assert key is not None and len(key) > 0
    assert re.match('^\w+$', key)
    return d.get(key)

@register.filter
def has_item(d, key):
    try:
        val = get_item(d, key)
        return True
    except KeyError:
        return False

@register.filter
def percentage(value):
    try:
        return value * 100.0
    except TypeError:
        return 0.0

@register.filter
@lru_cache(maxsize=1024)
def stock_sector(stock_code):
    assert stock_code is not None and len(stock_code) >= 3
    rec = CompanyDetails.objects.filter(asx_code=stock_code).first()
    if rec is None:
        return ''
    s = rec.sector_name
    return s