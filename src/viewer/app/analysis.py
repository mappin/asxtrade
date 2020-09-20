from app.models import Quotation, CompanyDetails, all_sector_stocks, desired_dates, company_prices
from datetime import datetime, timedelta
import pylru
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict

def as_css_class(thirty_day_slope, three_hundred_day_slope):
    if thirty_day_slope > 0.0 and three_hundred_day_slope < 0.0:
        return "recent-upward-trend"
    elif thirty_day_slope < 0.0 and three_hundred_day_slope > 0.0:
        return "recent-downward-trend"
    else:
        return "none"

def calculate_trends(cumulative_change_df, watchlist_stocks, all_dates):
    trends = {}  # stock -> (slope, nrmse) pairs
    for stock in watchlist_stocks:
        series = cumulative_change_df.loc[stock]
        n = len(series)
        series30 = series[-30:]
        coefficients, residuals, _, _, _ = np.polyfit(range(n), series, 1, full=True)
        coeff30, resid30, _, _, _ = np.polyfit(range(len(series30)), series30, 1, full=True)
        mse = residuals[0] / n
        nrmse = np.sqrt(mse) / (series.max() - series.min())
        if any([np.isnan(coefficients[0]), np.isnan(nrmse), abs(coefficients[0]) < 0.01 ]): # ignore stocks which are barely moving either way
            pass
        else:
            trends[stock] = (coefficients[0],
                             nrmse,
                             '{:.2f}'.format(coeff30[0]) if not np.isnan(coeff30[0]) else '',
                             as_css_class(coeff30[0], coefficients[0]))
    # sort by ascending overall slope (regardless of NRMSE)
    return OrderedDict(sorted(trends.items(), key=lambda t: t[1][0]))

def rank_cumulative_change(df, all_dates):
    cum_sum = defaultdict(float)
    for date in filter(lambda k: k in df.columns, all_dates):
        for code, price_change in df[date].fillna(0.0).iteritems():
            cum_sum[code] += price_change
        rank = pd.Series(cum_sum).rank(method='first', ascending=False)
        df[date] = rank

    all_available_dates = df.columns
    avgs = df.mean(axis=1) # NB: do this BEFORE adding columns...
    assert len(avgs) == len(df)
    df['x'] = all_available_dates[-1]
    df['y'] = df[all_available_dates[-1]]

    bins = ['top', 'bin2', 'bin3', 'bin4', 'bin5', 'bottom']
    average_rank_binned = pd.cut(avgs, len(bins), bins)
    assert len(average_rank_binned) == len(df)
    df['bin'] = average_rank_binned
    df['asx_code'] = df.index
    df['sector'] = [CompanyDetails.objects.get(asx_code=code).sector_name for code in df.index]
    df = pd.melt(df, id_vars=['asx_code', 'bin', 'sector', 'x', 'y'],
                     var_name='date',
                     value_name='rank',
                     value_vars=all_available_dates)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df['x'] = pd.to_datetime(df['x'], format="%Y-%m-%d")
    return df

def relative_strength(prices, n=14):
    # see https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    assert n > 0
    assert prices is not None and len(prices) >= n

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
    rsi.at[len(rsi)+1] = np.nan # ensure data series are the same length for matplotlib
    assert len(rsi) == len(prices)
    return rsi

def price_change_bins():
    """
    Return the bins and their label as a tuple for heatmap_market() to use and the
    plotting code
    """
    bins = [-1000.0, -100.0, -10.0, -5.0, -3.0, -2.0, -1.0, -1e-6, 0.0,
            1e-6, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0, 100.0, 1000.0]
    labels = ["{}".format(b) for b in bins[1:]]
    return (bins, labels)
