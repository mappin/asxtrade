from app.models import Quotation, CompanyDetails, all_sector_stocks, desired_dates, company_quotes, all_quotes
from datetime import datetime, timedelta
from pylru import lrudecorator
import pandas as pd
import numpy as np

def cumulative_change_summary(day, cum_price_change, start_prices, threshold=0.05): # 5% movement threshold to be considered momentum
    assert len(day) > 0
    assert isinstance(cum_price_change, dict)
    assert isinstance(start_prices, dict)

    n_pos = n_neg = n_unchanged = 0
    for code, cum_change in cum_price_change.items():
        trigger = threshold * start_prices[code]
        if cum_change > trigger:
            n_pos += 1
        elif cum_change < 0 and abs(cum_change) > trigger:
            n_neg += 1
        else:
            n_unchanged += 1
    return { 'date': day, 'n_pos': n_pos, 'n_neg': n_neg, 'n_unchanged': n_unchanged }

def analyse_companies(companies, n_days=90, initialisation_period=30, pct_trigger=5):
    """
    Return a pandas dataframe with n_days of data for the specified stock codes (all going well!)
    The initialisation period is used to get the numbers settled before the n_days begin.
    Returns a tuple (sector_df, best10, worst10) with the sector_df['date'] column as a pandas datetime instance
    """
    assert len(companies) > 0
    assert n_days > 0
    assert initialisation_period >= 0

    dates = desired_dates(n_days + initialisation_period)
    assert len(dates) >= n_days // 7 * 5 + initialisation_period // 7 * 5 - 10 # 10 is leeway for public holidays etc. (or missing data)
    print("Acquiring data for {} stocks over {} days".format(len(companies), len(dates)))

    aq = []
    start_prices = {}
    cum_price_change = {}
    t = pct_trigger / 100.0
    for day in sorted(dates, key=lambda k: datetime.strptime(k, "%Y-%m-%d")):
        for quote in company_quotes(companies, required_date=day):
            code = quote.asx_code
            if not code in start_prices:
                start_prices[code] = quote.last_price
                cum_price_change[code] = 0.0
            else:
                cum_price_change[code] += quote.change_price

        aq.append(cumulative_change_summary(day, cum_price_change,
                                                    start_prices, threshold=t))

    print("Found {} stocks with changes (expected {})".format(len(cum_price_change), len(companies)))
    pct_change = pd.Series(cum_price_change) / pd.Series(start_prices) * 100.0
    best10_pct = pct_change.nlargest(10)
    worst10_pct = pct_change.nsmallest(10)

    #print(aq[0])
    sector_df = pd.DataFrame.from_records(aq[initialisation_period:]) # skip data in initialisation period
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    #print(sector_df)
    return (sector_df, best10_pct, worst10_pct)

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

def heatmap_companies(companies, n_days=7):
    all_dates = desired_dates(n_days)
    queryset = all_quotes(companies, all_dates=all_dates)
    return heatmap_sentiment(queryset, all_dates)

@lrudecorator(2)
def heatmap_market(n_days=7):
    all_dates = desired_dates(n_days)
    queryset = all_quotes(None, all_dates=all_dates)
    return heatmap_sentiment(queryset, all_dates)

def heatmap_sentiment(queryset, all_dates):
    """
    Given a django queryset instance (required since further filtering will be performed)
    compute a dataframe ready for heatmapping of the quotations based using the percentage change on a daily basis.
    A tuple (dataframe, best10, worst10, n_stocks) is returned. n_days of data is used for all calculations.
    """
    assert queryset is not None
    assert all_dates is not None and len(all_dates) >= 1
    rows = []
    for date in all_dates:
        qs = queryset.all() # clone a fresh copy of queryset

        # although the supplied queryset might have filters, we ensure that we dont try to plot bad data
        quotes = qs.filter(fetch_date=date) \
                   .exclude(change_price__isnull=True) \
                   .exclude(error_code="id-or-code-invalid") \
                   .exclude(change_in_percent__isnull=True)
        for q in quotes:
            rows.append({ 'date': date, 'asx_code': q.asx_code, 'change_in_percent': q.percent_change() })
    df = pd.DataFrame.from_records(rows)
    #print(df)
    df = df.pivot(index='asx_code', columns='date', values='change_in_percent').fillna(0.0)
    assert len(df.columns) >= 4 # permit one public holiday, really expect 5
    n_stocks = len(df)
    top10 = {}
    bottom10 = {}
    bins, labels = price_change_bins()
    for date in df.columns:
        top10[date] = df[date].nlargest(10)
        bottom10[date] = df[date].nsmallest(10)
        df['bin_{}'.format(date)] = pd.cut(df[date], bins, labels=labels)
    # both bin assignments and raw values are returned in df
    return (df, top10, bottom10, n_stocks)
