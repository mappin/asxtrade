import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from app.analysis import *
import numpy as np
import pandas as pd
import base64
import io
from collections import Counter

def plot_as_base64(fig):
    """
    Convert supplied figure into string buffer and then return as base64-encoded data
    for insertion into a page as a context attribute
    """
    assert fig is not None
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64data = base64.b64encode(buf.read())
    return b64data

def make_sentiment_plot(sentiment_df, exclude_zero_bin=True, plot_text_labels=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), sharey=True)
    data = {}
    dates = []
    for column in filter(lambda c: c.startswith("bin_"), sentiment_df.columns):
        c = Counter(sentiment_df[column])
        date = column[4:]
        dates.append(date)
        data[date] = c
    df = pd.DataFrame.from_records(data)
    bins, labels = price_change_bins()
    df = df.reindex(reversed(labels))
    if exclude_zero_bin:
        df = df.drop('0.0')
    df = df.fillna(0)
    ax.imshow(df, aspect='auto')
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(dates)))
    ax.set_yticks(np.arange(len(sentiment_df)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(dates)
    bins = list(df.index)
    ax.set_yticklabels(bins)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if plot_text_labels:
        for i_idx, i in enumerate(bins):
            for j_idx, j in enumerate(dates):
               text = ax.text(j_idx, i_idx, int(df.iloc[i_idx, j_idx]),
                           ha="center", va="center", color="w")
    ax.grid(False)
    plt.plot()
    return plt.gcf()

def plot_heatmap(companies, all_dates=None, field_name='change_in_percent', bins=None):
    if bins is None:
        bins, labels = price_change_bins()
    if all_dates is None:
        all_dates = desired_dates(30)
    df = company_prices(companies, all_dates=all_dates, field_name=field_name) # by default change_in_percent will be used
    n_stocks = len(df)
    top10 = {}
    bottom10 = {}
    for date in df.columns:
        top10[date] = df[date].nlargest(10)
        bottom10[date] = df[date].nsmallest(10)
        df['bin_{}'.format(date)] = pd.cut(df[date], bins, labels=labels)
    fig = make_sentiment_plot(df, plot_text_labels=False)
    sentiment_data = plot_as_base64(fig).decode('utf-8')
    plt.close(fig)
    return (sentiment_data, df, top10, bottom10, n_stocks)

def make_sector_momentum_plot(dataframe, sector_name):
    assert len(sector_name) > 0
    assert len(dataframe) > 0

    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    timeline = dataframe['date']
    # now do the plot
    for name, ax, linecolour, title in zip(['n_pos', 'n_neg', 'n_unchanged'],
                                           axes,
                                           ['darkgreen', 'red', 'grey'],
                                           ['{} stocks up >5%'.format(sector_name), "{} stocks down >5%".format(sector_name), "Remaining stocks"]):
        # use a moving average to smooth out 5-day trading weeks and see the trend
        series = dataframe[name].rolling(14).mean()
        ax.plot(timeline, series, color=linecolour)
        ax.set_ylabel('', fontsize=8)
        ax.set_ylim(0, max(series.fillna(0))+10)
        ax.set_title(title, fontsize=8)

        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')
    plt.xticks(fontsize=8, rotation=30)
    plt.plot()
    ret = plt.gcf()
    data = plot_as_base64(ret).decode('utf-8')
    plt.close(fig)
    return data

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
    rsi = relative_strength(dataframe['last_price'])
    #print(len(rsi))
    fillcolor = 'darkgoldenrod'

    timeline = dataframe['fetch_date']
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
    fig = plt.gcf()
    rsi_data = plot_as_base64(fig).decode('utf-8')
    plt.close(fig)
    return rsi_data
