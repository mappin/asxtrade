from app.models import Quotation, CompanyDetails
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

# since companies is a list, it is unhashable, which means we cant use the decorator here
#@lrudecorator(10)
def analyse_companies(companies, n_days=90, initialisation_period=30, pct_trigger=5):
    """
    Return a pandas dataframe with n_days of data for the specified stock codes (all going well!)
    The initialisation period is used to get the numbers settled before the n_days begin.
    Returns a tuple (sector_df, best10, worst10) with the sector_df['date'] column as a pandas datetime instance
    """
    assert len(companies) > 0
    assert isinstance(companies[0], str) and len(companies[0]) >= 3
    assert n_days > 0
    assert initialisation_period >= 0

    start_date = datetime.today() - timedelta(days=n_days+initialisation_period) # extra 30 days for the numbers to settle
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, datetime.today())]
    assert len(dates) >= n_days // 7 * 5 + initialisation_period // 7 * 5 - 10 # 10 is leeway for public holidays etc. (or missing data)
    print("Acquiring data for {} stocks over {} days".format(len(companies), len(dates)))

    all_quotes = []
    start_prices = {}
    cum_price_change = {}
    t = pct_trigger / 100.0
    for day in sorted(dates, key=lambda k: datetime.strptime(k, "%Y-%m-%d")):
        daily_sector_quotes = Quotation.objects \
                                     .filter(fetch_date=day) \
                                     .filter(asx_code__in=companies) \
                                     .filter(change_price__isnull=False)

        for quote in daily_sector_quotes:
            code = quote.asx_code
            if not code in start_prices:
                start_prices[code] = quote.last_price
                cum_price_change[code] = 0.0
            else:
                cum_price_change[code] += quote.change_price

        all_quotes.append(cumulative_change_summary(day, cum_price_change,
                                                    start_prices, threshold=t))

    print("Found {} stocks with changes (expected {})".format(len(cum_price_change), len(companies)))
    pct_change = pd.Series(cum_price_change) / pd.Series(start_prices) * 100.0
    best10_pct = pct_change.nlargest(10)
    worst10_pct = pct_change.nsmallest(10)

    #print(all_quotes[0])
    sector_df = pd.DataFrame.from_records(all_quotes[initialisation_period:]) # skip data in initialisation period
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    #print(sector_df)
    return (sector_df, best10_pct, worst10_pct)

@lrudecorator(10)
def analyse_sector(sector_name, **kwargs):
    assert isinstance(sector_name, str) and len(sector_name) > 0
    sector = CompanyDetails.objects.filter(sector_name=sector_name)
    sector_stocks = [c.asx_code for c in sector]
    return analyse_companies(sector_stocks, **kwargs)

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

@lrudecorator(2)
def analyse_market(n_days=7):
    assert n_days > 0
    today = datetime.today()
    start_date = today - timedelta(days=n_days + 1) # +1 for today inclusive
    all_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, today)]
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
