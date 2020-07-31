from django.shortcuts import render
from app.models import Quotation, Security, CompanyDetails
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
from django.forms.models import model_to_dict

def all_stocks(request):
   # NB: dbfield is a str NOT date so order_by is just to get distinct working desirably
   all_dates = Quotation.objects.order_by('fetch_date').values_list('fetch_date', flat=True).distinct()
   # now we get the most recent date
   sorted_all_dates = sorted(all_dates, key=lambda k: datetime.strptime(k, "%Y-%m-%d"))
   assert len(sorted_all_dates) > 0
   ymd = sorted_all_dates[-1]
   assert isinstance(ymd, str) and len(ymd) > 8
   qs = Quotation.objects.order_by('-annual_dividend_yield', '-last_price', '-volume').filter(fetch_date=ymd).exclude(asx_code__isnull=True).exclude(last_price__isnull=True)
   assert qs is not None
   context = {
       "most_recent_date": ymd,
       "stocks": qs
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
    down = -seed[seed < 0].sum()/n
    rs = up/down
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
    for name, ax, linecolour in zip(['n_up', 'n_down', 'n_unchanged'], axes, ['darkgreen', 'red', 'grey']):
        # use a moving average to smooth out 5-day trading weeks and see the trend
        ax.plot(timeline, dataframe[name].rolling(7).mean(), color=linecolour)
        ax.set_ylabel('', fontsize=8)
        ax.set_title(name)

        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), fontsize=8)
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

    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.plot()
    return plt.gcf()

def as_dataframe(iterable):
    df = None
    headers = None
    for idx, rec in enumerate(iterable):
        d = model_to_dict(rec)
        if df is None:
            headers = d.keys()
            df = pd.DataFrame(columns=headers)
        df = df.append(d, ignore_index=True)
    if 'fetch_date' in headers:
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

@lrudecorator(10)
def analyse_sector(sector_name):
    assert isinstance(sector_name, str) and len(sector_name) > 0

    sector_stocks = [c.asx_code for c in CompanyDetails.objects.filter(sector_name=sector_name)]
    start_date = datetime.today() - timedelta(days=90)
    all_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, datetime.today())]
    assert len(all_dates) >= 80
    sector_df = pd.DataFrame(columns=['date', 'n_up', 'n_down', 'n_unchanged'])
    print("Found {} stocks in sector {}".format(len(sector_stocks), sector_name))
    for day in all_dates:
        daily_sector_quotes = Quotation.objects. \
                                     filter(fetch_date=day). \
                                     filter(asx_code__in=sector_stocks). \
                                     filter(change_price__isnull=False)
        pos = neg = zeroes = 0
        for quote in daily_sector_quotes:
            if quote.change_price > 0.0:
                pos += 1
            elif quote.change_price < 0.0:
                neg += 1
            else:
                zeroes += 1
        sector_df = sector_df.append({ 'date': day, 'n_up': pos,
                                       'n_down': neg, 'n_unchanged': zeroes },
                                     ignore_index=True)
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    return sector_df

def show_stock(request, stock=None):
   stock_regex = re.compile('^\w+$')
   assert stock_regex.match(stock)

   # make stock momentum plot
   quotes = Quotation.objects.filter(asx_code=stock)
   securities = Security.objects.filter(asx_code=stock)
   company_details = CompanyDetails.objects.filter(asx_code=stock).first()
   df = as_dataframe(quotes)
   print(df['last_price'])
   assert len(df) > 0
   fig = make_rsi_plot(stock, df)
   rsi_data = plot_as_base64(fig)

   # show sector performance over past 3 months
   sector_df = analyse_sector(company_details.sector_name)
   #print(sector_df)
   fig = make_sector_momentum_plot(sector_df)
   sector_b64 = plot_as_base64(fig)

   # populate template and render HTML page with context
   context = {
       'rsi_data': rsi_data.decode('utf-8'),
       'asx_code': stock,
       'securities': securities,
       'cd': company_details,
       'sector_past3months_data': sector_b64.decode('utf-8'),
   }
   return render(request, "stock_view.html", context=context)
