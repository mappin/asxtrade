from django import forms
from django.core.exceptions import ValidationError
from app.models import CompanyDetails
from pylru import lrudecorator

def is_not_blank(value):
    if value == None or len(value) < 1 or len(value.strip()) < 1:
        raise ValidationError("Invalid value - cannot be blank")

class SectorSearchForm(forms.Form):
    @lrudecorator(1)
    def asx_sectors():
       all_sectors = list(CompanyDetails.objects.mongo_distinct('sector_name'))
       results = [(sector, sector) for sector in all_sectors]
       return results

    sector = forms.ChoiceField(choices=asx_sectors, required=True, validators=[is_not_blank])


class DividendSearchForm(forms.Form):
    min_yield = forms.FloatField(required=False, min_value=0.0)
    max_yield = forms.FloatField(required=False, min_value=0.0, max_value=1000.0)

class CompanySearchForm(forms.Form):
    name = forms.CharField(required=False)
    activity = forms.CharField(required=False)
