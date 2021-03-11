from django import forms
from django.core.exceptions import ValidationError
from app.models import all_sector_stocks
from django.db import models

def is_not_blank(value):
    if value is None or len(value) < 1 or len(value.strip()) < 1:
        raise ValidationError("Invalid value - cannot be blank")

def is_valid_sector(value):
    assert value is not None
    return len(all_sector_stocks(value)) > 0

class SectorSearchForm(forms.Form):
    SECTOR_CHOICES = (
        ("Class Pend", "Class Pend"),
        ("Communication Services", "Communication Services"),
        ("Consumer Discretionary", "Consumer Discretionary"),
        ("Consumer Staples", "Consumer Staples"),
        ("Energy", "Energy"),
        ("Financials", "Financials"),
        ("Health Care", "Health Care"),
        ("Industrials", "Industrials"),
        ("Information Technology", "Information Technology"),
        ("Materials", "Materials"),
        ("Metals & Mining", "Metals & Mining"),
        ("Not Applic", "Not Applic"),
        ("Real Estate", "Real Estate"),
        ("Utilities", "Utilities"),
    )
    sector = forms.ChoiceField(choices=SECTOR_CHOICES, required=True, validators=[is_not_blank, is_valid_sector])
    best10 = forms.BooleanField(required=False, label="Best 10 performers (past 3 months)")
    worst10 = forms.BooleanField(required=False, label="Worst 10 performers (past 3 months)")

class DividendSearchForm(forms.Form):
    min_yield = forms.FloatField(required=False, min_value=0.0, initial=4.0)
    max_yield = forms.FloatField(required=False, min_value=0.0, max_value=1000.0, initial=100.0)
    min_pe = forms.FloatField(required=False, min_value=0.0, initial=0.0, label="Min P/E")
    max_pe = forms.FloatField(required=False, max_value=1000.0, initial=30.0, label="Max P/E")
    min_eps_aud = forms.FloatField(required=False, min_value=-1000.0, initial=0.01, label="Min EPS ($AUD)")

class CompanySearchForm(forms.Form):
    name = forms.CharField(required=False)
    activity = forms.CharField(required=False)

class MoverSearchForm(forms.Form):
    threshold = forms.FloatField(required=True, min_value=0.0, max_value=10000.0, initial=50.0)
    timeframe_in_days = forms.IntegerField(required=True, min_value=1, max_value=365, initial=7, label="Timeframe (days)")
    show_increasing = forms.BooleanField(required=False, initial=True, label="Increasing")
    show_decreasing = forms.BooleanField(required=False, initial=True, label="Decreasing")

class SectorSentimentSearchForm(forms.Form):
    normalisation_choices = (
        (1, 'None'),
        (2, 'Min/Max. Scaler'),
        (3, 'Divide by Max')
    )
    sector = forms.ChoiceField(required=True, choices=SectorSearchForm.SECTOR_CHOICES)
    normalisation_method = forms.ChoiceField(required=True, choices=normalisation_choices)
