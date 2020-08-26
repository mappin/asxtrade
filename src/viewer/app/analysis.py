from app.models import Quotation, CompanyDetails, all_sector_stocks, desired_dates, company_prices
from datetime import datetime, timedelta
import pylru
import pandas as pd
import numpy as np

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
            1e-6, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0, 1000.0]
    labels = ["{}".format(b) for b in bins[1:]]
    return (bins, labels)
