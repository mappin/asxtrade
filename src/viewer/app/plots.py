import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import plotnine as p9
from app.analysis import company_prices
from app.models import stocks_by_sector, desired_dates, day_low_high
import numpy as np
import pandas as pd
import base64
import io
from datetime import datetime
from collections import Counter, defaultdict


def price_change_bins():
    """
    Return the bins and their label as a tuple for heatmap_market() to use and the
    plotting code
    """
    bins = [-1000.0, -100.0, -10.0, -5.0, -3.0, -2.0, -1.0, -1e-6, 0.0,
            1e-6, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0, 100.0, 1000.0]
    labels = ["{}".format(b) for b in bins[1:]]
    return (bins, labels)

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
    rows = []
    print("Sentiment plot: exclude zero bins? {} show text? {}".format(exclude_zero_bin, plot_text_labels))

    for column in filter(lambda c: c.startswith("bin_"), sentiment_df.columns):
        c = Counter(sentiment_df[column])
        date = column[4:]
        for bin, val in c.items():
            if exclude_zero_bin and (bin == '0.0' or not isinstance(bin, str)):
                continue
            bin = str(bin)
            assert isinstance(bin, str)
            val = int(val)
            rows.append({ 'date': datetime.strptime(date, "%Y-%m-%d"), 'bin': bin, 'value': val })

    df = pd.DataFrame.from_records(rows)
    #print(df['bin'].unique())
    # HACK TODO FIXME: should get from price_change_bins()...
    order = ['-1000.0', '-100.0', '-10.0', '-5.0', '-3.0', '-2.0', '-1.0', '-1e-06',
             '1e-06', '1.0', '2.0', '3.0', '5.0', '10.0', '25.0', '100.0', '1000.0']
    df['bin_ordered'] = pd.Categorical(df['bin'], categories=order)

    plot = (p9.ggplot(df, p9.aes('date', 'bin_ordered', fill='value'))
            + p9.geom_tile(show_legend=False) + p9.theme_bw()
            + p9.xlab("") + p9.ylab("Percentage daily change")
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=7), figure_size=(10,5)))
    if plot_text_labels:
        plot = plot + p9.geom_text(p9.aes(label='value'), size=8, color="white")
    return plot_as_inline_html_data(plot)

def plot_key_stock_indicators(df, stock):
    assert isinstance(df, pd.DataFrame)
    assert all(['eps' in df.columns, 'pe' in df.columns, 'annual_dividend_yield' in df.columns])

    df['volume'] = df['last_price'] * df['volume'] / 1000000 # again, express as $(M)
    df['fetch_date'] = df.index
    plot_df = pd.melt(df, id_vars='fetch_date',
                 value_vars=['pe', 'eps', 'annual_dividend_yield', 'volume', 'last_price'],
                 var_name='indicator',
                 value_name='value')
    plot_df['value'] = pd.to_numeric(plot_df['value'])
    plot_df['fetch_date'] = pd.to_datetime(plot_df['fetch_date'])

    plot = (p9.ggplot(plot_df, p9.aes('fetch_date', 'value', color='indicator'))
            + p9.geom_line(size=1.5, show_legend=False)
            + p9.facet_wrap('~ indicator', nrow=6, ncol=1, scales='free_y')
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=7), figure_size=(8,7))
        #    + p9.aes(ymin=0)
            + p9.xlab("") + p9.ylab("")
    )
    return plot_as_inline_html_data(plot)

def plot_as_inline_html_data(plot, charset='utf-8'):
    """
    Return utf-8 encoded base64 image data for inline insertion into HTML content
    using the template engine. Plot must be a valid plotnine ggplot instance (or compatible)
    This function performs all required cleanup of the figure state, so callers can be clean.
    """
    assert plot is not None
    fig = plot.draw()
    data = plot_as_base64(fig).decode(charset)
    plt.close(fig)
    return data

def plot_portfolio(portfolio_df, figure_size=(12, 4), line_size=1.5, date_text_size=7):
    """
    Given a daily snapshot of virtual purchases plot both overall and per-stock
    performance. Return a tuple of figures representing the performance as inline data.
    """
    assert portfolio_df is not None
    #print(portfolio_df)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    avg_profit_over_period = portfolio_df.filter(items=['stock', 'stock_profit']).groupby('stock').mean()
    avg_profit_over_period['contribution'] = ['positive' if profit >= 0.0 else 'negative' for profit in avg_profit_over_period.stock_profit]
    avg_profit_over_period = avg_profit_over_period.drop('stock_profit', axis='columns') # dont want to override actual profit with average
    portfolio_df = portfolio_df.merge(avg_profit_over_period, left_on='stock', right_index=True, how='inner')
    #print(portfolio_df)

    # 1. overall performance
    df = portfolio_df.filter(items=['portfolio_cost', 'portfolio_worth', 'portfolio_profit', 'date'])
    df = df.melt(id_vars=['date'], var_name='field')
    plot = (p9.ggplot(df, p9.aes('date', 'value', group='field', color='field'))
            + p9.labs(x='', y='$ AUD')
            + p9.geom_line(size=1.5)
            + p9.facet_wrap('~ field', nrow=3, ncol=1, scales='free_y')
            + p9.theme(axis_text_x=p9.element_text(angle=30, size=date_text_size),
                       figure_size=figure_size,
                       legend_position='none')
    )
    overall_figure = plot_as_inline_html_data(plot)

    df = portfolio_df.filter(items=['stock', 'date', 'stock_profit', 'stock_worth', 'contribution'])
    melted_df = df.melt(id_vars=['date', 'stock', 'contribution'], var_name='field')
    all_dates = sorted(melted_df['date'].unique())
    df = melted_df[melted_df['date'] == all_dates[-1]]
    df = df[df['field'] == 'stock_profit'] # only latest profit is plotted
    df['contribution'] = ['positive' if profit >= 0.0 else 'negative' for profit in df['value']]

    # 2. plot contributors ie. winners and losers
    plot = (p9.ggplot(df, p9.aes('stock', 'value', fill='stock'))
            + p9.geom_bar(stat='identity')
            + p9.labs(x='', y='$ AUD')
            + p9.facet_grid('contribution ~ field')
            + p9.theme(legend_position='none', figure_size=figure_size)
    )
    profit_contributors = plot_as_inline_html_data(plot)

    # 3. per purchased stock performance
    plot = (p9.ggplot(melted_df, p9.aes('date', 'value', group='stock', colour='stock'))
            + p9.xlab('')
            + p9.geom_line(size=1.0)
            + p9.facet_grid('field ~ contribution', scales="free_y")
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=date_text_size),
                       figure_size=figure_size,
                       panel_spacing=0.5,  # more space between plots to avoid tick mark overlap
                       subplots_adjust = {'right': 0.8})
    )
    stock_figure = plot_as_inline_html_data(plot)
    return overall_figure, stock_figure, profit_contributors

def plot_company_rank(df):
    assert isinstance(df, pd.DataFrame)
    #assert 'sector' in df.columns
    n_bin = len(df['bin'].unique())
    plot = (p9.ggplot(df, p9.aes('date', 'rank', group='asx_code', color='sector'))
            + p9.geom_smooth(span=0.3, se=False)
            + p9.geom_text(p9.aes(label='asx_code', x='x', y='y'), nudge_x=1.2, size=6, show_legend=False)
            + p9.xlab('')
            + p9.facet_wrap('~bin', nrow=n_bin, ncol=1, scales="free_y")
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=7),
                       figure_size=(8, 20),
                       subplots_adjust={'right': 0.8})
           )
    return plot_as_inline_html_data(plot)

def plot_company_versus_sector(df, stock, sector):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(stock, str)
    assert isinstance(sector, str)
    df['date'] = pd.to_datetime(df['date'])
    #print(df)
    plot = (p9.ggplot(df, p9.aes('date', 'value', group='group', color='group', fill='group'))
            + p9.geom_line(size=1.5)
            + p9.xlab('')
            + p9.ylab('Percentage change since start')
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=7),
                       figure_size=(8,4),
                       subplots_adjust={'right': 0.8})
    )
    return plot_as_inline_html_data(plot)

def plot_market_wide_sector_performance(all_dates, field_name='change_in_percent'):
    """
    Display specified dates for average sector performance. Each company is assumed to have at zero
    at the start of the observation period. A plot as base64 data is returned.
    """
    df = company_prices(None, all_dates=all_dates, fields='change_in_percent') # None == all stocks
    n_stocks = len(df)
    # merge in sector information for each company
    code_and_sector = stocks_by_sector()
    n_unique_sectors = len(code_and_sector['sector_name'].unique())
    print("Found {} unique sectors".format(n_unique_sectors))

    #print(code_and_sector)
    df = df.merge(code_and_sector, left_on='asx_code', right_on='asx_code')
    print("Found {} stocks, {} sectors and merged total: {}".format(n_stocks, len(code_and_sector), len(df)))
    # compute average change in percent of each unique sector over each day and sum over the dates
    cumulative_pct_change = df.expanding(axis='columns').sum()
    # merge date-wise into df
    for date in cumulative_pct_change.columns:
        df[date] = cumulative_pct_change[date]
    #df.to_csv('/tmp/crap.csv')
    grouped_df = df.groupby('sector_name').mean()
    #grouped_df.to_csv('/tmp/crap.csv')

    # ready the dataframe for plotting
    grouped_df = pd.melt(grouped_df, ignore_index=False, var_name='date', value_name='cumulative_change_percent')
    grouped_df['sector'] = grouped_df.index
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])
    n_col = 3
    plot = (p9.ggplot(grouped_df, p9.aes('date', 'cumulative_change_percent', color='sector'))
            + p9.geom_line(size=1.0)
            + p9.facet_wrap('~sector', nrow=n_unique_sectors // n_col + 1, ncol=n_col, scales='free_y')
            + p9.xlab('')
            + p9.ylab('Average sector change (%)')
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=6),
                       axis_text_y = p9.element_text(size=6),
                       figure_size = (12, 6),
                       panel_spacing=0.3,
                       legend_position='none')
    )
    return plot_as_inline_html_data(plot)

def plot_series(df, x=None, y=None, tick_text_size=6, line_size=1.5,
                y_axis_label='Point score', x_axis_label=''):
    assert len(df) > 0
    assert len(x) > 0 and len(y) > 0
    assert line_size > 0.0
    assert isinstance(tick_text_size, int) and tick_text_size > 0
    assert y_axis_label is not None
    assert x_axis_label is not None
    plot = (p9.ggplot(df, p9.aes(x=x, y=y, color='stock'))
            + p9.geom_line(size=line_size)
            + p9.labs(x=x_axis_label, y=y_axis_label)
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=tick_text_size),
                       axis_text_y = p9.element_text(size=tick_text_size),
                       legend_position='none')
    )
    return plot_as_inline_html_data(plot)

def plot_heatmap(companies, all_dates=None, field_name='change_in_percent', bins=None, n_top_bottom=10):
    """
    Using field_name plot a data matrix as a heatmap by change_in_percent using the specified bins. If not
    using change_in_percent as the field name, you may need to adjust the bins to the values being used.
    The horizontal axis is the dates specified - past 30 days by default. Also computes top10/worst10 and
    returns a tuple (plot, dataframe, top10, bottom10, n_stocks). Top10/Bottom10 will contain n_top_bottom items.
    """
    if bins is None:
        bins, labels = price_change_bins()
    if all_dates is None:
        all_dates = desired_dates(start_date=30)
    df = company_prices(companies, all_dates=all_dates, fields=field_name) # by default change_in_percent will be used
    n_stocks = len(df)
    sum = df.sum(axis=1) # compute totals across all dates for the specified companies to look at performance across the observation period
    top10 = sum.nlargest(n_top_bottom)
    bottom10 = sum.nsmallest(n_top_bottom)
    #print(df.columns)
    #print(bins)
    try:
        # NB: this may fail if no prices are available so we catch that error and handle accordingly...
        for date in df.columns:
            df['bin_{}'.format(date)] = pd.cut(df[date], bins, labels=labels)
        sentiment_plot = make_sentiment_plot(df, plot_text_labels=len(all_dates) <= 21) # show counts per bin iff not too many bins
        return (sentiment_plot, df, top10, bottom10, n_stocks)
    except KeyError:
        return (None, None, None, None, None)

def plot_sector_performance(dataframe, descriptor, window_size=14):
    assert len(descriptor) > 0
    assert len(dataframe) > 0

    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    timeline = pd.to_datetime(dataframe['date'])
    locator, formatter = auto_dates()
    # now do the plot
    for name, ax, linecolour, title in zip(['n_pos', 'n_neg', 'n_unchanged'],
                                           axes,
                                           ['darkgreen', 'red', 'grey'],
                                           ['{} stocks up >5%'.format(descriptor), "{} stocks down >5%".format(descriptor), "Remaining stocks"]):
        # use a moving average to smooth out 5-day trading weeks and see the trend
        series = dataframe[name].rolling(window_size).mean()
        ax.plot(timeline, series, color=linecolour)
        ax.set_ylabel('', fontsize=8)
        ax.set_ylim(0, max(series.fillna(0))+10)
        ax.set_title(title, fontsize=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        # Remove the automatic x-axis label from all but the bottom subplot
        if ax != axes[-1]:
            ax.set_xlabel('')
    plt.plot()
    ret = plt.gcf()
    data = plot_as_base64(ret).decode('utf-8')
    plt.close(fig)
    return data

def auto_dates():
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ['%y',  # ticks are mostly years
                         '%b',       # ticks are mostly months
                         '%d',       # ticks are mostly days
                         '%H:%M',    # hrs
                         '%H:%M',    # min
                         '%S.%f', ]  # secs
    # these are mostly just the level above...
    formatter.zero_formats = [''] + formatter.formats[:-1]
    # ...except for ticks that are mostly hours, then it is nice to have
    # month-day:
    formatter.zero_formats[3] = '%d-%b'

    formatter.offset_formats = ['',
                                '%Y',
                                '%b %Y',
                                '%d %b %Y',
                                '%d %b %Y',
                                '%d %b %Y %H:%M', ]
    return (locator, formatter)

def relative_strength(prices, n=14):
    # see https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    assert n > 0
    assert prices is not None

    # Get the difference in price from previous step
    delta = prices.diff()

    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=n).mean()
    roll_down1 = down.abs().ewm(span=n).mean()

    # Calculate the RSI based on EWMA
    rs = roll_up1 / roll_down1
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # NB: format is carefully handled here, so downstream code doesnt break
    new_date = datetime.strftime(datetime.now(), "%Y-%m-%d ") # make sure it is not an existing date
    #print(new_date)
    rsi.at[new_date] = np.nan # ensure data series are the same length for matplotlib
    #print(len(rsi), " ", len(prices))
    #assert len(rsi) == len(prices)
    return rsi

def make_rsi_plot(stock, stock_df):
    assert len(stock) > 0

    #print(last_price)
    #print(volume)
    #print(day_low_price)
    #print(day_high_price)

    last_price = stock_df['last_price']
    volume = stock_df['volume']
    day_low_price = stock_df['day_low_price']
    day_high_price = stock_df['day_high_price']

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
    fig.autofmt_xdate()

    # plot the relative strength indicator
    rsi = relative_strength(last_price)
    #print(len(rsi))
    fillcolor = 'darkgoldenrod'

    timeline = pd.to_datetime(last_price.index)
    #print(values)
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
    #ax1.set_title('{} daily'.format(stock))

    # plot the price and volume data
    dx = 0.0
    low = day_low_price + dx
    high = day_high_price + dx

    deltas = np.zeros_like(last_price)
    deltas[1:] = np.diff(last_price)
    up = deltas > 0
    ax2.vlines(timeline[up], low[up], high[up], color='black', label='_nolegend_')
    ax2.vlines(timeline[~up], low[~up], high[~up], color='black', label='_nolegend_')
    ma20 = last_price.rolling(window=20).mean()
    ma200 = last_price.rolling(window=200).mean()

    #timeline = timeline.to_list()
    linema20, = ax2.plot(timeline, ma20, color='blue', lw=2, label='MA (20)')
    linema200, = ax2.plot(timeline, ma200, color='red', lw=2, label='MA (200)')
    assert linema20 is not None
    assert linema200 is not None

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

    volume = (last_price * volume)/1e6  # dollar volume in millions
    #print(volume)
    vmax = max(volume)
    poly = ax2t.fill_between(timeline, volume.to_list(), 0, alpha=0.5,
                             label='Volume', facecolor=fillcolor, edgecolor=fillcolor)
    assert poly is not None # avoid unused variable from pylint
    ax2t.set_ylim(0, 5*vmax)
    ax2t.set_yticks([])

    # compute the MACD indicator
    fillcolor = 'darkslategrey'

    n_fast = 12
    n_slow = 26
    n_ema= 9
    emafast = last_price.ewm(span=n_fast, adjust=False).mean()
    emaslow = last_price.ewm(span=n_slow, adjust=False).mean()
    macd = emafast - emaslow
    nema = macd.ewm(span=n_ema, adjust=False).mean()
    ax3.plot(timeline, macd, color='black', lw=2)
    ax3.plot(timeline, nema, color='blue', lw=1)
    ax3.fill_between(timeline, macd - nema, 0, alpha=0.3, facecolor=fillcolor, edgecolor=fillcolor)
    ax3.text(0.025, 0.95, 'MACD ({}, {}, {})'.format(n_fast, n_slow, n_ema), va='top',
             transform=ax3.transAxes, fontsize=textsize)

    ax3.set_yticks([])
    locator, formatter = auto_dates()
    for ax in ax1, ax2, ax2t, ax3:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    plt.xticks(fontsize=8)
    fig = plt.gcf()
    rsi_data = plot_as_base64(fig).decode('utf-8')
    plt.close(fig)
    return rsi_data

def plot_best_monthly_price_trend(dataframe, field='open_price'):
    assert dataframe is not None
    assert field in dataframe.columns
    assert 'fetch_date' in dataframe.columns
    dataframe = dataframe.filter(items=['fetch_date', field], axis='columns')
    dataframe = dataframe.set_index(pd.to_datetime(dataframe['fetch_date']))
    dataframe = dataframe.resample('M', kind='period').max()
    plot = (p9.ggplot(dataframe, p9.aes(x='dataframe.index', y=field))
            + p9.geom_bar(stat='identity', fill='#880000', alpha=0.5)
            + p9.labs(x='', y='$AUD')
            + p9.theme(axis_text_x = p9.element_text(angle=30, size=7))
           )
    return plot_as_inline_html_data(plot)

def plot_point_scores(stock: str, sector_companies, 
                      all_stocks_cip: pd.DataFrame,
                      rules):
    """
    Visualise the stock in terms of point scores as described on the stock view page. Rules to apply
    can be specified by rules (default rules are provided by rule_*())

    Points are lost for equivalent downturns and the result plotted. All rows in all_stocks_cip will be
    used to calculate the market average on a given trading day, whilst only sector_companies will
    be used to calculate the sector average. A utf-8 base64 encoded plot image is returned
    """
    assert len(stock) >= 3
    assert all_stocks_cip is not None
    assert rules is not None and len(rules) > 0

    rows = []
    points = 0
    day_low_high_df = day_low_high(stock, all_dates=all_stocks_cip.columns)
    state = { 'day_low_high_df': day_low_high_df,  # never changes each day, so we init it here
              'all_stocks_change_in_percent_df': all_stocks_cip,
              'stock': stock,
              'daily_range_threshold': 0.20, # 20% at either end of the daily range gets a point
            }
    net_points_by_rule = defaultdict(int)
    for date in all_stocks_cip.columns:
        market_avg = all_stocks_cip[date].mean()
        sector_avg = all_stocks_cip[date].filter(items=sector_companies).mean()
        stock_move = all_stocks_cip.at[stock, date]
        state.update({ 'market_avg': market_avg, 'sector_avg': sector_avg,
                       'stock_move': stock_move, 'date': date })
        points += sum(map(lambda r: r(state), rules))
        for r in rules:
            k = r.__name__
            if k.startswith("rule_"):
                k = k[5:]
            net_points_by_rule[k] += r(state)
        rows.append({ 'points': points, 'stock': stock, 'date': date })

    df = pd.DataFrame.from_records(rows)
    df['date'] = pd.to_datetime(df['date'])
    point_score_plot = plot_series(df, x='date', y='points')

    rows = []
    for k,v in net_points_by_rule.items():
        rows.append({ 'rule': str(k), 'net_points': v})
    df = pd.DataFrame.from_records(rows)
    net_rule_contributors_plot = (p9.ggplot(df, p9.aes(x='rule', y='net_points'))
            + p9.labs(x="Rule", y="Contribution to points by rule")
            + p9.geom_bar(stat="identity")
            + p9.theme(axis_text_y = p9.element_text(size=7),
                       subplots_adjust={'left': 0.2})
            + p9.coord_flip()
    )
    return point_score_plot, plot_as_inline_html_data(net_rule_contributors_plot)