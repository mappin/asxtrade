"""
Responsible for interaction with worldbank data API and interaction with the user
"""
import re
import io
from collections import defaultdict
from typing import Iterable
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import FormView
from django.http import Http404
from django.urls import reverse
from django.shortcuts import render
import plotnine as p9
from mizani.formatters import date_format
import pandas as pd
from app.data import cache_plot
from worldbank.models import WorldBankIndicators, WorldBankCountry, WorldBankTopic, WorldBankInvertedIndex
from worldbank.forms import WorldBankSCSMForm, WorldBankSCMForm, WorldBankSCMMForm


# BUGGY TODO FIXME: doesnt handle multiple-metric view well and choice of significant digits sucks also
def label_shorten(labels: Iterable[float]) -> Iterable[str]:
    """
    For large numbers we report them removing all the zeros and replacing with billions/millions/trillions as appropriate.
    This is only done if all labels in the plot end with zeros and all are sufficiently long to have enough zeros. Otherwise labels are untouched.
    """
    str_labels = []
    non_zero_labels = []
    for v in labels:
        magnitude_v = abs(v)
        new_str = str(int(v)) if magnitude_v > 100000.0 else "{:f}".format(v)
        # trim trailing zeros so it doesnt confuse the code below?
        if re.match(r'^.*\.0+$', new_str):
            new_str = re.sub(r'\.0+$', '', new_str)
            # fallthru...

        str_labels.append(new_str)  # NB: some plots have negative value tick marks, so we must handle it
        if magnitude_v > 1e-10:
            non_zero_labels.append(new_str)

    #print(str_labels)
    #print(non_zero_labels)
    non_zero_labels_set = set(non_zero_labels)

    for units, short_label in [("000000000000", "T."), ("000000000", "B."), ("000000", "M.")]:
        found_cnt = 0
        for l in non_zero_labels:
            if l.endswith(units):
                found_cnt += 1
        if found_cnt == len(non_zero_labels):
            new_labels = []
            for old_label in str_labels:
                #print(old_label)
                #print(units)
                if old_label in non_zero_labels_set:
                    base_label = old_label[0:-len(units)]
                    new_label = base_label + short_label
                    new_labels.append(new_label)
                else:
                    new_labels.append(old_label)
            return new_labels
    # no way to consistently shorten labels? ok, return supplied labels
    return labels

@login_required
def worldbank_index_view(request):
    return render(request, 'world_bank.html', context={
        "topics":     WorldBankTopic.objects.all().order_by('topic'),
        "indicators": WorldBankIndicators.objects.all(),
        "countries":  WorldBankCountry.objects.all().order_by('name')
    })

def indicator_autocomplete_hits(country_codes: Iterable[str], topic_id: int, as_select_tuples=False) -> tuple:
    assert country_codes is not None
    assert topic_id is not None
    assert len(country_codes) > 0
    assert re.match(r'^\d+$', topic_id)
    for c in country_codes:
        assert re.match(r'^\w{2,4}$', c)
    topic_id = int(topic_id)

    if len(country_codes) == 1: # optimisation to avoid using __in and the performance hit
        records = WorldBankInvertedIndex.objects.filter(topic_id=topic_id, country=next(iter(country_codes))) # cannot index a set, so we use next() instead
    else:
        country_set = set(country_codes) # ensure no dupes
        n_countries = len(country_set)
        #print(f"Topic ID: {topic_id}")
        print(f"Country set: {country_set}")
        records = WorldBankInvertedIndex.objects.filter(topic_id=topic_id, country__in=country_set)
        # show only those indicators FOR WHICH ALL SPECIFIED COUNTRIES are available
        countries_by_indicator = defaultdict(set)
        for r in records:
            countries_by_indicator[r.indicator_id].add(r.country)
        print("Countries for each metric: {}".format(countries_by_indicator))
        acceptable_indicators = set(filter(lambda t: len(countries_by_indicator[t]) == n_countries, countries_by_indicator.keys()))
        #print("Acceptable indicators: {}".format(acceptable_indicators))
        records = records.filter(indicator_id__in=acceptable_indicators)

    hits = []
    seen = set()
    for i in sorted(records, key=lambda i: i.indicator_name):
        key = i.indicator_id
        if key in seen:
            continue
        seen.add(key)
        hits.append(i)
    print("Found {} matching indicators".format(len(hits)))
    current_id = None
    if len(hits) > 0:
        current_id = hits[0].indicator_id
    if as_select_tuples:
        hits = [(i.indicator_id, i.indicator_name) for i in hits]
    return hits, current_id


def worldbank_plot(df: pd.DataFrame, title: str, dates_are_yearly: bool, figure_size=(12,6), add_points=False, **plot_kwargs) -> p9.ggplot:
    """
    Carefully written to support all worldbank plots, this method is the one place where the app needs themes, colour maps
    and various plot related settings. For sparse datasets it used geom_point() in addition to geom_line() in case the data
    is so sparse that lines cannot be drawn. Returns a ggplot instance or raises an exception if the dataframe is empty.
    """
    if df is None:
        print(f"No usable data/plot for {title}")
        raise Http404(f"No data for {title}")

    pct_na = (df['metric'].isnull().sum() / len(df)) * 100.0
    assert pct_na >= 0.0 and pct_na <= 100.0
    
    plot = ( p9.ggplot(df, p9.aes("date", "metric", **plot_kwargs))
            + p9.geom_path(size=1.2) 
            + p9.theme_classic()
            + p9.labs(x="", y="Value", title=title)
            + p9.theme(figure_size=figure_size)
            + p9.scale_y_continuous(labels=label_shorten)
            + p9.scale_color_cmap_d()
    )
    if dates_are_yearly:
        plot += p9.scale_x_datetime(labels=date_format('%Y'))  # yearly data? if so only print the year on the x-axis
    # if pct_na is too high, geom_path() may be unable to draw a line (each value is surrounded by nan preventing a path)
    # so we use geom_point() to highlight the sparse nature of the data
    if pct_na >= 30.0 or add_points or df['metric'].count() <= 3:
        plot += p9.geom_point(size=3.0)
    return plot

def fetch_data(indicator: WorldBankIndicators, country_names: Iterable[str], fill_missing=None) -> pd.DataFrame:
    """
    Fetch data from the market_data_cache collection (not to be confused with the market_quote_cache collection)
    and ensure the specified countries are only present in the data (if present). Optionally apply a callable to
    fill in gaps eg. resample
    """
    if indicator is None:
        return None
    with io.BytesIO(indicator.fetch_data()) as fp:
        df = pd.read_parquet(fp)
        if df is not None and len(df) > 0:
            plot_df = df[df["country"].isin(country_names)]
            #print(country_names)
            if len(plot_df) == 0:
                return None
            if fill_missing:
                #print(plot_df)
                plot_df.index = pd.to_datetime(plot_df['date'], format='%Y-%m-%d') # avoid callers having to specify 'on' keyword as they may not know which column
                plot_df = fill_missing(plot_df)
                #print(plot_df)
            return plot_df
        else:
            return None

def get_countries():
    """Only return those countries with country codes that appear in the inverted index ie. have datasets. Ordered tuples are returned suitable
    for a HTML select form element. Tuples are ordered by printable country name. """
    country_codes_with_datasets = set(WorldBankInvertedIndex.objects.all().values_list('country', flat=True).distinct())
    return [(c.country_code, c.name) for c in WorldBankCountry.objects.filter(country_code__in=country_codes_with_datasets).order_by('name')]

def ajax_autocomplete_view(request):
    assert request.method == 'GET'
    topic_id = request.GET.get('topic_id', None)
    country_code = request.GET.get('country', None)
    hits, current_id = indicator_autocomplete_hits([country_code], topic_id)
    return render(request, "autocomplete_indicators.html", context={ 'hits': hits, "current_id": current_id })

def ajax_autocomplete_scm_view(request):
    assert request.method == 'GET'
    topic_id = request.GET.get('topic_id', None)
    country_codes = set(filter(lambda v: len(v) > 0, request.GET.get('country', '').split(",")))
    
    if len(country_codes) > 0:
        hits, current_id = indicator_autocomplete_hits(country_codes, topic_id)
    else:
        hits = []
        current_id = None
    return render(request, "autocomplete_indicators.html", context={ 'hits': hits, "current_id": current_id })

class WorldBankSCSMView(LoginRequiredMixin, FormView): # single-country, single metric (indicator) the easiest to start with
    action_url = '/worldbank/scsm'
    form_class = WorldBankSCSMForm
    template_name = "world_bank_scsm.html"

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        countries_as_tuples = get_countries()
        topics_as_tuples = [(t.id, t.topic) for t in WorldBankTopic.objects.all().order_by('topic')]
        kwargs = self.get_form_kwargs()
        data = kwargs.get('data', {})
        selected_country = data.get('country', None)
        selected_topic = data.get('topic', None)
        if selected_topic is not None and selected_country is not None:
            indicators_as_tuples, _ = indicator_autocomplete_hits([selected_country], selected_topic, as_select_tuples=True)
        else:
            indicators_as_tuples = []
        return form_class(countries_as_tuples, topics_as_tuples, indicators_as_tuples, **kwargs)

    def plot_indicator(self, country: WorldBankCountry, topic: WorldBankTopic, indicator: WorldBankIndicators) -> str:
        def make_plot():
            df = fetch_data(indicator, [country.name], 
                            fill_missing=lambda df: df.resample('AS').asfreq())
            return worldbank_plot(df, indicator.name, True)

        return cache_plot(f"{indicator.wb_id}-{country.name}-scsm-worldbank-plot", make_plot)
            
    def form_valid(self, form):
        #print(form.cleaned_data)
        country = form.cleaned_data.get('country', None)
        topic = form.cleaned_data.get('topic', None)
        indicator = form.cleaned_data.get('indicator', None)
        if country is None or topic is None:
            raise Http404('Both country and topic must be specified for this plot')
        country = WorldBankCountry.objects.filter(country_code=country).first()
        topic = WorldBankTopic.objects.filter(id=topic).first()
        if topic is None:
            raise Http404('No such topic: -{}-'.format(topic))
        
        indicator = WorldBankIndicators.objects.filter(wb_id=indicator).first()
        context = self.get_context_data()
        context.update({
            "title": "World Bank data - single country, single metric",
            "country": country,
            "topic": topic,
            "indicator_autocomplete_uri": reverse('ajax-worldbank-scsm-autocomplete'),
            "indicator_plot_uri": self.plot_indicator(country, topic, indicator),
            "indicator_plot_title": f"Graphic: {indicator.name}",
            "indicator": indicator
        })
        return render(self.request, self.template_name, context=context)

class WorldBankSCMView(LoginRequiredMixin, FormView):
    action_url = "/worldbank/scm"
    form_class = WorldBankSCMForm
    template_name = "world_bank_scm.html"

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        countries_as_tuples = get_countries()
        topics_as_tuples = [(t.id, t.topic) for t in WorldBankTopic.objects.all().order_by('topic')]
        kwargs = self.get_form_kwargs()
        data = kwargs.get('data', {})
        selected_countries = data.get('countries', '').split(",")
        selected_topic = data.get('topic', None)
        
        if selected_countries is not None and selected_topic is not None:
            assert isinstance(selected_countries, list)
            indicators_as_tuples, _ = indicator_autocomplete_hits(selected_countries, selected_topic, as_select_tuples=True)
        else:
            indicators_as_tuples = []
        return form_class(countries_as_tuples, topics_as_tuples, indicators_as_tuples, **kwargs)

    def plot_country_comparison(self, countries: Iterable[str], topic: WorldBankTopic, indicator: WorldBankIndicators) -> p9.ggplot:
        def fix_gaps(df: pd.DataFrame) -> pd.DataFrame:
            """
            If say the plot timeframe is between 1960 and 2020 but there are missing rows where some years used to be,
            this method will re-introduce new rows into the dataframe and set the country column to the specified country. It must
            only be called when len(countries) == 1 or a pandas error will occur
            """
            df = df.resample('AS').asfreq()
            df['country'] = next(iter(countries))  # NB: this is only correct wen len(countries) == 1
            return df

        def make_plot():
            resample_lambda = None
            if len(countries) == 1:
                resample_lambda = fix_gaps
            df = fetch_data(indicator, countries, fill_missing=resample_lambda) # not resampling to fill gaps at this time, unless only one country is being plotted: TODO BUG FIXME
            kwargs = { 'group': "country", 'colour': "country"}
            #print(df)
            return worldbank_plot(df, indicator.name, True, **kwargs)

        countries_str = "-".join(countries)
        return cache_plot(f"{indicator.wb_id}-{countries_str}-scm-worldbank-plot", make_plot, dont_cache=True)

    def form_valid(self, form):
        selected_countries = form.cleaned_data.get('countries', None)
        selected_topic = form.cleaned_data.get('topic', None)
        selected_indicator = form.cleaned_data.get('indicator', None)

        topic = WorldBankTopic.objects.filter(id=selected_topic).first()
        if topic is None:
            raise Http404('No such topic: -{}-'.format(topic))
        countries = WorldBankCountry.objects.filter(country_code__in=selected_countries)
        country_names = [c.name for c in countries]
        indicator = WorldBankIndicators.objects.filter(wb_id=selected_indicator).first()
        context = self.get_context_data()
        context.update({
            "title": "World Bank data - single metric over selected countries",
            "countries": selected_countries,
            "topic": topic,
            "indicator_autocomplete_uri": reverse('ajax-worldbank-scm-autocomplete'),
            "indicator_plot_uri": self.plot_country_comparison(country_names, topic, indicator),
            "indicator_plot_title": f"Graphic: {indicator.name}",
            "indicator": indicator
        })
        return render(self.request, self.template_name, context=context)

class WorldBankSCMMView(LoginRequiredMixin, FormView):
    action_url = "/worldbank/scmm"
    template_name = "world_bank_scmm.html"
    form_class = WorldBankSCMMForm

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        countries_as_tuples = get_countries()
        topics_as_tuples = [(t.id, t.topic) for t in WorldBankTopic.objects.all().order_by('topic')]
        kwargs = self.get_form_kwargs()
        data = kwargs.get('data', {})
        selected_country = data.get('country', None)
        selected_topic = data.get('topic', None)
    
        if selected_country is not None and selected_topic is not None:
            indicators_as_tuples, _ = indicator_autocomplete_hits([selected_country], selected_topic, as_select_tuples=True)
        else:
            indicators_as_tuples = []
        #print(indicators_as_tuples)
        return form_class(countries_as_tuples, topics_as_tuples, indicators_as_tuples, **kwargs)

    def plot_multiple_metrics(self, country: str, topic: WorldBankTopic, indicators: Iterable[WorldBankIndicators]) -> p9.ggplot:
        def make_plot():
            plot_df = None
            has_yearly = False
            n_datasets = 0
            add_points = False
            for i in indicators:
                df = fetch_data(i, [country], fill_missing=lambda df: df.resample('AS').asfreq())
                if df is None or len(df) == 0:
                    continue
                n_datasets += 1
                df['dataset'] = f"{i.name} ({i.wb_id})"
                if "-yearly-" in i.tag:
                    has_yearly = True
                pct_na = (df['metric'].isnull().sum() / len(df)) * 100.0
               
                if pct_na > 30.0 or df['metric'].count() <= 3:
                    add_points = True
                if plot_df is None:
                    plot_df = df
                else:
                    # if any indicator is sparse, we enable points for all indicators to be able to see them all
                    plot_df = plot_df.append(df)

            #print(plot_df)
            figure_size = (12, n_datasets * 1.5)
            kwargs = { 'group': "dataset", 'colour': "dataset" }
            plot = worldbank_plot(plot_df, '', has_yearly, figure_size=figure_size, add_points=add_points, **kwargs)
            plot += p9.facet_wrap('~dataset', ncol=1, scales="free_y")
            plot += p9.theme(legend_position='none') # disable legend since each facet has its own title
            return plot

        indicator_id_str = "-".join([i.wb_id for i in indicators])
        return cache_plot(f"{country}-{indicator_id_str}-scmm-worldbank-plot", make_plot, dont_cache=True)

    def form_valid(self, form):
        selected_country = form.cleaned_data.get('country', None)
        selected_topic = form.cleaned_data.get('topic', None)
        selected_indicators = form.cleaned_data.get('indicators', [])
        #print(selected_indicators)

        topic = WorldBankTopic.objects.filter(id=selected_topic).first()
        if topic is None:
            raise Http404('No such topic: -{}-'.format(topic))
        country = WorldBankCountry.objects.filter(country_code=selected_country).first()
        indicators = WorldBankIndicators.objects.filter(wb_id__in=set(selected_indicators))
        n_indicators = len(indicators)
        context = self.get_context_data()
        context.update({
            "title": "World Bank data - multiple metrics over single country",
            "country": selected_country,
            "topic": topic,
            "indicator_autocomplete_uri": reverse('ajax-worldbank-scmm-autocomplete'),
            "indicator_plot_uri": self.plot_multiple_metrics(country.name, topic, indicators),
            "indicator_plot_title": f"Graphic: {n_indicators} metrics for {country.name}",
            "indicator_plot_datasets": indicators,
        })
        return render(self.request, self.template_name, context=context)