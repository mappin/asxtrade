from django.template.defaulttags import register
import re


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