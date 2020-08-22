from django.shortcuts import render, get_object_or_404
from django.forms.models import model_to_dict
from django.views.generic import FormView, TemplateView
from django.http import HttpResponseRedirect, Http404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic.list import MultipleObjectTemplateResponseMixin, MultipleObjectMixin
from app.models import *
from app.forms import SectorSearchForm, DividendSearchForm, CompanySearchForm
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import io
import re
import base64
import urllib
import numpy as np
import pandas as pd
from pylru import lrudecorator


class SearchMixin:
    paginate_by = 50
    model = Quotation
    object_list = Quotation.objects.none()

    def get(self, request, *args, **kwargs):
       # need to subclass this method to ensure pagination works correctly (as 'next', 'last' etc. is GET not POST)
       d = {}
       key = self.__class__.__name__
       print("Updating session state: {}".format(key))
       d.update(request.session.get(key, {})) # update the form to the session state
       return self.update_form(d)

    def update_form(self, form_values):
       assert isinstance(form_values, dict)
       # apply the form settings to self.queryset (specific to a CBV - watch for subclass overrides)
       self.object_list = self.get_queryset(**form_values)
       state_field = self.__class__.__name__  # NB: must use class name so that each search type has its own state for a given user
       self.request.session[state_field] = form_values
       context = self.get_context_data()
       assert context is not None
       assert self.action_url is not None
       context['action_url'] = self.action_url
       self.form = self.form_class(initial=form_values)
       context['form'] = self.form
       return self.render_to_response(context)

    def form_invalid(self, form):
       return self.update_form(form.cleaned_data)

    # this is called from self.post()
    def form_valid(self, form):
       assert form.is_valid()
       return self.update_form(form.cleaned_data)

class SectorSearchView(SearchMixin, LoginRequiredMixin, MultipleObjectMixin, MultipleObjectTemplateResponseMixin, FormView):
    form_class = SectorSearchForm
    template_name = "search_form.html" # generic template, not specific to this view
    action_url = '/search/by-sector'
    ordering = ('-annual_dividend_yield', 'asx_code') # keep pagination happy, but not used by get_queryset()
    sector_name = None
    as_at_date = None

    def render_to_response(self, context):
        context.update({
            'stocks': self.object_list,
            'sector': self.sector_name,
            'most_recent_date': self.as_at_date,
            'watched': user_watchlist(self.request.user)
        })
        return super().render_to_response(context)

    def get_queryset(self, **kwargs):
       if kwargs == {}:
           return Quotation.objects.none()
       assert 'sector' in kwargs and len(kwargs['sector']) > 0
       self.sector_name = kwargs['sector']
       all_available_dates = sorted(Quotation.objects.mongo_distinct('fetch_date'),
                                    key=lambda k: datetime.strptime(k, "%Y-%m-%d"))
       wanted_companies = CompanyDetails.objects.filter(sector_name=self.sector_name)
       wanted_stocks = [wc.asx_code for wc in wanted_companies]
       self.as_at_date = all_available_dates[-1]
       print("Looking for {} companies as at {}".format(len(wanted_stocks), self.as_at_date))
       results = Quotation.objects.filter(fetch_date=self.as_at_date) \
                                  .filter(asx_code__in=wanted_stocks) \
                                  .order_by(*self.ordering)
       return results

sector_search = SectorSearchView.as_view()

class DividendYieldSearch(SearchMixin, LoginRequiredMixin, MultipleObjectMixin, MultipleObjectTemplateResponseMixin, FormView):
    form_class = DividendSearchForm
    template_name = "search_form.html" # generic template, not specific to this view
    action_url = '/search/by-yield'
    ordering = ('-annual_dividend_yield', 'asx_code') # keep pagination happy, but not used by get_queryset()
    as_at_date = None

    def render_to_response(self, context):
        context.update({
           'stocks': self.object_list,
           'most_recent_date': self.as_at_date,
           'watched': user_watchlist(self.request.user)
        })
        return super().render_to_response(context)

    def get_queryset(self, **kwargs):
       if kwargs == {}:
           return Quotation.objects.none()

       self.as_at_date = latest_quotation_date()
       min_yield = kwargs.get('min_yield') if 'min_yield' in kwargs else 0.0
       max_yield = kwargs.get('max_yield') if 'max_yield' in kwargs else 10000.0
       results = Quotation.objects.filter(fetch_date=self.as_at_date) \
                                  .filter(annual_dividend_yield__gte=min_yield) \
                                  .filter(annual_dividend_yield__lte=max_yield) \
                                  .order_by(*self.ordering)
       return results

dividend_search = DividendYieldSearch.as_view()


class CompanySearch(DividendYieldSearch):
    form_class = CompanySearchForm
    action_url = "/search/by-company"

    def get_queryset(self, **kwargs):
        if kwargs == {} or not any(['name' in kwargs, 'activity' in kwargs]):
            return Quotation.objects.none()

        self.as_at_date = latest_quotation_date()
        matching_companies = set()
        wanted_name = kwargs.get('name', '')
        wanted_activity = kwargs.get('activity', '')
        if len(wanted_name) > 0:
            # match by company name first...
            matching_companies.update([hit.asx_code for hit in
                                       CompanyDetails.objects.filter(name_full__icontains=wanted_name)])
            # but also matching codes
            matching_companies.update([hit.asx_code for hit in
                                       CompanyDetails.objects.filter(asx_code__icontains=wanted_name)])
        if len(wanted_activity) > 0:
            matching_companies.update([hit.asx_code for hit in \
                                       CompanyDetails.objects.filter(principal_activities__icontains=wanted_activity)])
        print("Showing results for {} companies".format(len(matching_companies)))
        results = Quotation.objects.filter(fetch_date=self.as_at_date) \
                                   .filter(asx_code__in=matching_companies) \
                                   .order_by(*self.ordering)
        return results

company_search = CompanySearch.as_view()

@login_required
def all_stocks(request):
   # NB: dbfield is a str NOT date so order_by is just to get distinct working desirably
   all_dates = Quotation.objects.order_by('fetch_date').values_list('fetch_date', flat=True).distinct()
   # now we get the most recent date
   sorted_all_dates = sorted(all_dates, key=lambda k: datetime.strptime(k, "%Y-%m-%d"))
   assert len(sorted_all_dates) > 0
   ymd = sorted_all_dates[-1]
   assert isinstance(ymd, str) and len(ymd) > 8
   qs = Quotation.objects.order_by('-annual_dividend_yield', '-last_price', '-volume') \
                         .filter(fetch_date=ymd) \
                         .exclude(asx_code__isnull=True) \
                         .exclude(last_price__isnull=True)
   assert qs is not None
   context = {
       "most_recent_date": ymd,
       "stocks": qs,
       "watched": user_watchlist(request.user)
   }
   return render(request, "all_stocks.html", context=context)

def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum() / n
    if abs(down) < 1e-6 :
        raise Http404("Relative strength not available due to no prices")
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

def make_sector_momentum_plot(dataframe):
    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    timeline = dataframe['date']
    # now do the plot
    for name, ax, linecolour, title in zip(['n_pos', 'n_neg', 'n_unchanged'],
                                           axes,
                                           ['darkgreen', 'red', 'grey'],
                                           ['Stocks up over 5% in past 90 days', "Stocks down over 5% in past 90 days", "Remaining stocks"]):
        # use a moving average to smooth out 5-day trading weeks and see the trend
        ax.plot(timeline, dataframe[name].rolling(30).mean(), color=linecolour)
        ax.set_ylabel('', fontsize=8)
        ax.set_title(name)

        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')
    plt.xticks(fontsize=8, rotation=30)
    plt.plot()
    return plt.gcf()

def make_rsi_plot(stock_code, dataframe):
    plt.rc('axes', grid=True)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

    textsize = 8
    left, width = 0.1, 0.8
    rect1 = [left, 0.7, width, 0.2]
    rect2 = [left, 0.3, width, 0.4]
    rect3 = [left, 0.1, width, 0.2]

    fig = plt.figure(facecolor='white')
    axescolor = '#f6f6f6'  # the axes background color

    ax1 = fig.add_axes(rect1, facecolor=axescolor)  # left, bottom, width, height
    ax2 = fig.add_axes(rect2, facecolor=axescolor, sharex=ax1)
    ax2t = ax2.twinx()
    ax3 = fig.add_axes(rect3, facecolor=axescolor, sharex=ax1)

    # plot the relative strength indicator
    prices = pd.to_numeric(dataframe['last_price'], errors='coerce').to_numpy()
    rsi = relative_strength(prices)
    fillcolor = 'darkgoldenrod'

    timeline = dataframe.fetch_date
    ax1.plot(timeline, rsi, color=fillcolor)
    ax1.axhline(70, color='darkgreen')
    ax1.axhline(30, color='darkgreen')
    ax1.fill_between(timeline, rsi, 70, where=(rsi >= 70), facecolor=fillcolor, edgecolor=fillcolor)
    ax1.fill_between(timeline, rsi, 30, where=(rsi <= 30), facecolor=fillcolor, edgecolor=fillcolor)
    ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
    ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([30, 70])
    ax1.text(0.025, 0.95, 'RSI (14)', va='top', transform=ax1.transAxes, fontsize=textsize)
    #ax1.set_title('{} daily'.format(stock_code))

    # plot the price and volume data
    dx = 0.0
    low = dataframe.day_low_price + dx
    high = dataframe.day_high_price + dx

    deltas = np.zeros_like(prices)
    deltas[1:] = np.diff(prices)
    up = deltas > 0
    ax2.vlines(timeline[up], low[up], high[up], color='black', label='_nolegend_')
    ax2.vlines(timeline[~up], low[~up], high[~up], color='black', label='_nolegend_')
    ma20 = dataframe.last_price.rolling(window=20).mean()
    ma200 = dataframe.last_price.rolling(window=200).mean()

    timeline = timeline.to_list()
    linema20, = ax2.plot(timeline, ma20, color='blue', lw=2, label='MA (20)')
    linema200, = ax2.plot(timeline, ma200, color='red', lw=2, label='MA (200)')

    #last = dataframe[-1]
    #s = '%s O:%1.2f H:%1.2f L:%1.2f C:%1.2f, V:%1.1fM Chg:%+1.2f' % (
    #    today.strftime('%d-%b-%Y'),
    #    last.open, last.high,
    #    last.low, last.close,
    #    last.volume*1e-6,
    #    last.close - last.open)
    #t4 = ax2.text(0.3, 0.9, s, transform=ax2.transAxes, fontsize=textsize)

    props = font_manager.FontProperties(size=10)
    leg = ax2.legend(loc='center left', shadow=True, fancybox=True, prop=props)
    leg.get_frame().set_alpha(0.5)

    volume = (prices * dataframe.volume)/1e6  # dollar volume in millions
    #print(volume)
    vmax = max(volume)
    poly = ax2t.fill_between(timeline, volume.to_list(), 0, alpha=0.5,
                             label='Volume', facecolor=fillcolor, edgecolor=fillcolor)
    ax2t.set_ylim(0, 5*vmax)
    ax2t.set_yticks([])

    # compute the MACD indicator
    fillcolor = 'darkslategrey'

    n_fast = 12
    n_slow = 26
    n_ema= 9
    emafast = dataframe.last_price.ewm(span=n_fast, adjust=False).mean()
    emaslow = dataframe.last_price.ewm(span=n_slow, adjust=False).mean()
    macd = emafast - emaslow
    nema = macd.ewm(span=n_ema, adjust=False).mean()
    ax3.plot(timeline, macd, color='black', lw=2)
    ax3.plot(timeline, nema, color='blue', lw=1)
    ax3.fill_between(timeline, macd - nema, 0, alpha=0.3, facecolor=fillcolor, edgecolor=fillcolor)
    ax3.text(0.025, 0.95, 'MACD ({}, {}, {})'.format(n_fast, n_slow, n_ema), va='top',
             transform=ax3.transAxes, fontsize=textsize)

    ax3.set_yticks([])
    # turn off upper axis tick labels, rotate the lower ones, etc
    for ax in ax1, ax2, ax2t, ax3:
        if ax != ax3:
            for label in ax.get_xticklabels():
                label.set_visible(False)
        else:
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment('right')

        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    plt.xticks(fontsize=8)
    return plt.gcf()

def as_dataframe(iterable):
    rows = []
    for rec in iterable:
        d = model_to_dict(rec)
        rows.append(d)
    df = pd.DataFrame.from_records(rows)
    if 'fetch_date' in df.columns:
        df['fetch_date'] = pd.to_datetime(df['fetch_date'])
        df = df.sort_values(by='fetch_date')
    return df

def plot_as_base64(fig):
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64data = base64.b64encode(buf.read())
    return b64data

def assess_cumulative_change(day, cum_price_change, start_prices, threshold=0.05): # 5% threshold is considered momentum
    assert len(day) > 0
    assert isinstance(cum_price_change, dict)
    assert isinstance(start_prices, dict)

    n_pos = n_neg = n_unchanged = 0
    for code, cum_change in cum_price_change.items():
        price_threshold = threshold * start_prices[code]
        if cum_change > price_threshold:
            n_pos += 1
        elif cum_change < 0 and abs(cum_change) > price_threshold:
            n_neg += 1
        else:
            n_unchanged += 1
    return { 'date': day, 'n_pos': n_pos, 'n_neg': n_neg, 'n_unchanged': n_unchanged }

@lrudecorator(10)
def analyse_sector(sector_name, initialisation_period_in_days=30):
    assert isinstance(sector_name, str) and len(sector_name) > 0

    sector = CompanyDetails.objects.filter(sector_name=sector_name)
    sector_stocks = [c.asx_code for c in sector]
    start_date = datetime.today() - timedelta(days=90+initialisation_period_in_days) # extra 30 days for the numbers to settle
    all_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, datetime.today())]
    assert len(all_dates) >= 80 + initialisation_period_in_days * 0.8
    print("Found {} stocks in sector {}".format(len(sector_stocks), sector_name))
    all_quotes = []
    start_prices = {}
    cum_price_change = {}
    for day in sorted(all_dates, key=lambda k: datetime.strptime(k, "%Y-%m-%d")):
        daily_sector_quotes = Quotation.objects. \
                                     filter(fetch_date=day). \
                                     filter(asx_code__in=sector_stocks). \
                                     filter(change_price__isnull=False)

        for quote in daily_sector_quotes:
            code = quote.asx_code
            if not code in start_prices:
                start_prices[code] = quote.last_price
                cum_price_change[code] = 0.0
            else:
                cum_price_change[code] += quote.change_price

        all_quotes.append(assess_cumulative_change(day, cum_price_change, start_prices))

    sector_df = pd.DataFrame.from_records(all_quotes[initialisation_period_in_days:]) # skip initialisation period
    #print(sector_df)
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    return sector_df

def update_image_cache(tag, base64_data, valid_days=7): # cache for a week by default
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

def show_stock(request, stock=None):
   validate_stock(stock)

   # make stock momentum plot
   quotes = Quotation.objects.filter(asx_code=stock).exclude(last_price__isnull=True)
   securities = Security.objects.filter(asx_code=stock)
   company_details = CompanyDetails.objects.filter(asx_code=stock).first()
   if company_details is None:
       raise Http404("No company details for {}".format(stock))
   df = as_dataframe(quotes)
   #print(df['last_price'])
   if len(df) == 0:
       raise Http404("No price quotes for {}".format(stock))
   fig = make_rsi_plot(stock, df)
   rsi_data = plot_as_base64(fig)
   plt.close(fig)

   # show sector performance over past 3 months
   tag = "sector_momentum-{}".format(company_details.sector_name)
   cache_hit = ImageCache.objects.filter(tag=tag).first()
   if cache_hit is None or cache_hit.is_outdated():
       sector_df = analyse_sector(company_details.sector_name)
       #print(sector_df)
       fig = make_sector_momentum_plot(sector_df)
       sector_b64 = plot_as_base64(fig).decode('utf-8')
       plt.close(fig)
       update_image_cache(tag, sector_b64)
   else:
       sector_b64 = cache_hit.base64

   # populate template and render HTML page with context
   context = {
       'rsi_data': rsi_data.decode('utf-8'),
       'asx_code': stock,
       'securities': securities,
       'cd': company_details,
       'sector_past3months_data': sector_b64,
   }
   return render(request, "stock_view.html", context=context)

@lrudecorator(2)
def analyse_market(start_date):
    all_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, datetime.today())]
    assert len(all_dates) == 7
    df = pd.DataFrame(columns=['date', 'asx_code', 'change_in_percent'])
    all_quotes = {}
    all_stocks = set()
    for date in all_dates:
        quotes = Quotation.objects.filter(fetch_date=date) \
                                  .exclude(change_price__isnull=True) \
                                  .exclude(error_code='id-or-code-invalid') \
                                  .exclude(change_in_percent__isnull=True)
        #print("Obtained {} quotes for {}".format(len(quotes), date))
        daily_quotes = { q.asx_code: q.percent_change() for q in quotes }
        for_plotting = {}
        for k,v in daily_quotes.items():
            if abs(v) > 1.0: # exclude boring movements from the plot
                for_plotting[k] = v
                all_stocks.add(k)
        if len(for_plotting.keys()) > 10: # ignore days where nothing happens
            all_quotes[date] = for_plotting

    # every pandas data series must have the same keys in it or a plotting error will occur
    all_series = []
    for date in filter(lambda k: k in all_quotes, all_dates):
        d = all_quotes[date]
        assert isinstance(d, dict)
        for k in all_stocks:
            if not k in d:
                d.update({ k: 0 })
        series = pd.Series(all_quotes[date], name=date)
        all_series.append(series)

    return all_series

def make_market_sentiment_plot(data_series):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=True)
    bp = ax.violinplot(data_series, widths=0.9)
    labels = [ds.name for ds in data_series]
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    plt.plot()
    return plt.gcf()

def market_sentiment(request):
    start_date = datetime.today() - timedelta(days=6)
    all_data_series = analyse_market(start_date)
    fig = make_market_sentiment_plot(all_data_series)
    sentiment_data = plot_as_base64(fig)
    plt.close(fig)
    top10 = { series.name: series.nlargest(n=10) for series in all_data_series }
    bottom10 = { series.name: series.nsmallest(n=10) for series in all_data_series }
    #print(top10)
    context = {
       'sentiment_data': sentiment_data.decode('utf-8'),
       'n_days': len(all_data_series),
       'n_stocks_plotted': max([len(series) for series in all_data_series]),
       'best_ten': top10,
       'worst_ten': bottom10,
    }
    return render(request, 'market_sentiment_view.html', context=context)

@login_required
def show_watched(request):
    matching_companies = user_watchlist(request.user)

    as_at = latest_quotation_date()
    print("Showing results for {} companies".format(len(matching_companies)))
    # here we use a dict to give to the template: so we can augment user_purchases
    # into the template ie. not just model objects given to the template. Doing
    # it this way makes it easier to display purchases in the template
    results = { hit.asx_code: model_to_dict(hit) for hit in Quotation.objects.filter(fetch_date=as_at) \
                               .filter(asx_code__in=matching_companies) \
                               .order_by('asx_code') }
    purchases = user_purchases(request.user)
    for key, stock in results.items():
        if key in purchases:
            assert isinstance(stock, dict)
            stock['virtual_purchases'] = purchases[key]
    #print(results.values())
    context = {
         "most_recent_date": as_at,
         "stocks": results.values(),
         "title": "Stocks you are watching",
         "watched": user_watchlist(request.user),
    }
    return render(request, 'all_stocks.html', context=context)

def validate_stock(stock):
    assert stock is not None
    assert isinstance(stock, str) and len(stock) >= 3
    assert re.match('^\w+$', stock)

def validate_date(d):
    assert isinstance(d, str) and len(d) < 20  # YYYY-mm-dd must be less than 20
    assert re.match('^\d{4}-\d{2}-\d{2}$', d)

def redirect_to_next(request, fallback_next='/'):
    # redirect will trigger a redraw which will show the purchase since next will be the same page
    assert request is not None
    if request.GET is not None:
        return HttpResponseRedirect(request.GET.get('next', fallback_next))
    else:
        return HttpResponseRedirect(fallback_next)

@login_required
def toggle_watched(request, stock=None):
    validate_stock(stock)
    current_watchlist = user_watchlist(request.user)
    if stock in current_watchlist: # remove from watchlist?
        Watchlist.objects.filter(user=request.user, asx_code=stock).delete()
    else:
        w = Watchlist(user=request.user, asx_code=stock)
        w.save()
    return redirect_to_next(request)

@login_required
def buy_virtual_stock(request, stock=None, amount=5000.0):
    validate_stock(stock)
    assert amount > 0.0
    cur_price, latest_date = latest(stock)
    if cur_price >= 1e-6:
        vp = VirtualPurchase(asx_code=stock, user=request.user,
                             buy_date=latest_date, price_at_buy_date=cur_price,
                             amount=amount, n=int(amount / cur_price))
        vp.save()
        print("Purchased {} as at {}: {} shares at {} each".format(vp.asx_code, latest_date, vp.n, vp.price_at_buy_date))
    else:
        print("WARNING: cant buy {} as its price is zero".format(stock))
        # do nothing in this case...
        pass
    return redirect_to_next(request)

@login_required
def delete_virtual_stock(request, buy_date=None, stock=None):
    validate_stock(stock)
    validate_date(buy_date)
    print(buy_date)
    print(stock)
    stocks_to_delete = VirtualPurchase.objects.filter(buy_date=buy_date, asx_code=stock)
    print("Selected {} stocks to delete".format(len(stocks_to_delete)))
    stocks_to_delete.delete()
    return redirect_to_next(request)
