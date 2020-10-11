from app.models import Quotation, CompanyDetails, all_sector_stocks, company_prices, day_low_high
from app.plots import *
from app.messages import warning
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

def rule_move_up(state: dict):
    """
    Return 1 if the state indicates the stock moved up, 0 otherwise
    """
    assert state is not None

    move = state.get('stock_move')
    if move > 0.0:
       return 1
    return 0

def rule_market_avg(state: dict):
    """
    If the magnitude of the stock move is greater than the magnitude of the
    market move, then award two points in the direction of the move. Otherwise 0
    """
    assert state is not None

    move = state.get('stock_move')
    market_avg = state.get('market_avg')
    if abs(move) >= abs(market_avg):
        return np.sign(move) * 2
    return 0

def rule_sector_avg(state: dict):
    """
    Award 3 points for a stock beating the sector average on a given day or
    detract three points if it falls more than the magnitude of the sector
    """
    assert state is not None

    move = state.get('stock_move')
    sector_avg = state.get('sector_avg')

    if abs(move) >= abs(sector_avg):
        return np.sign(move) * 3
    return 0

def rule_signif_move(state: dict):
    """
    Only ~10% of stocks will move by more than 2% percent on a given day, so give a point for that...
    """
    assert state is not None

    move = state.get('stock_move') # percentage move
    if move >= 2.0:
        return 1
    elif move <= -2.0:
        return -1
    return 0

def rule_against_market(state: dict):
    """
    Award 1 point (in the direction of the move) if the stock moves against
    the overall market AND sector in defiance of the global sentiment (eg. up on a down day)
    """
    assert state is not None
    stock_move = state.get('stock_move')
    market_avg = state.get('market_avg')
    sector_avg = state.get('sector_avg')
    if stock_move > 0.0 and market_avg < 0.0 and sector_avg < 0.0:
        return 1
    elif stock_move < 0.0 and market_avg > 0.0 and sector_avg > 0.0:
        return -1
    return 0

def rule_at_end_of_daily_range(state: dict):
    """
    Award 1 point if the price at the end of the day is within 20% of the daily trading range (either end)
    Otherwise 0.
    """
    assert state is not None
    day_low_high_df = state.get('day_low_high_df')
    date = state.get('date')
    threshold = state.get('daily_range_threshold')
    try:
        day_low = day_low_high_df.at[date, 'day_low_price']
        day_high = day_low_high_df.at[date, 'day_high_price']
        last_price = day_low_high_df.at[date, 'last_price']
        assert not np.isnan(day_low) and not np.isnan(day_high)
        range = (day_high - day_low) * threshold # 20% at either end of daily range
        if last_price >= day_high - range:
            return 1
        elif last_price <= day_low + range:
            return -1
        # else FALLTHRU...
    except KeyError:
        stock = state.get('stock')
        warning(None, "Unable to obtain day low/high and last_price for {} on {}".format(stock, date))
    return 0

def default_point_score_rules():
    """
    Return a list of rules to apply as a default list during analyse_point_scores()
    """
    return [rule_move_up,
            rule_market_avg,
            rule_sector_avg,
            rule_signif_move,
            rule_against_market,
            rule_at_end_of_daily_range
            ]

def detect_outliers(stocks: list, all_stocks_cip: pd.DataFrame, rules=None):
    """
    Returns a dataframe describing those outliers present in stocks based on the provided rules.
    """
    if rules is None:
        rules = default_point_score_rules()
    str_rules = { str(r):r for r in rules }
    rows = []
    stocks_by_sector_df = stocks_by_sector() # NB: ETFs in watchlist will have no sector
    stocks_by_sector_df.index = stocks_by_sector_df['asx_code']
    for stock in stocks:
        #print("Processing stock: ", stock)
        try:
           sector = stocks_by_sector_df.at[stock, 'sector_name']
           sector_companies = list(stocks_by_sector_df.loc[stocks_by_sector_df['sector_name'] == sector].asx_code)
           # day_low_high() may raise KeyError when data is currently being fetched, so it appears here...
           day_low_high_df = day_low_high(stock, all_stocks_cip.columns)
        except KeyError:
           warning(None, "Unable to locate watchlist entry: {} - continuing without it".format(stock))
           continue
        state = {
            'day_low_high_df': day_low_high_df,  # never changes each day, so we init it here
            'all_stocks_change_in_percent_df': all_stocks_cip,
            'stock': stock,
            'daily_range_threshold': 0.20, # 20% at either end of the daily range gets a point
        }
        points_by_rule = defaultdict(int)
        for date in all_stocks_cip.columns:
            market_avg = all_stocks_cip[date].mean()
            sector_avg = all_stocks_cip[date].filter(items=sector_companies).mean()
            stock_move = all_stocks_cip.at[stock, date]
            state.update({ 'market_avg': market_avg, 'sector_avg': sector_avg,
                           'stock_move': stock_move, 'date': date })
            for rule_name, rule in str_rules.items():
                points_by_rule[rule_name] += rule(state)
        d = { 'stock': stock }
        d.update(points_by_rule)
        rows.append(d)
    df = pd.DataFrame.from_records(rows)
    df = df.set_index('stock')
    print(df)
    from pyod.models.iforest import IForest
    clf = IForest()
    clf.fit(df)
    scores = clf.predict(df)
    results = [row[0] for row, value in zip(df.iterrows(), scores) if value > 0]
    #print(results)
    print("Found {} outlier stocks".format(len(results)))
    return results

def analyse_point_scores(stock: str, sector_companies, all_stocks_cip: pd.DataFrame, rules=None):
    """
    Visualise the stock in terms of point scores as described on the stock view page. Rules to apply
    can be specified by rules (default rules are provided by rule_*())

    Points are lost for equivalent downturns and the result plotted. All rows in all_stocks_cip will be
    used to calculate the market average on a given trading day, whilst only sector_companies will
    be used to calculate the sector average. A utf-8 base64 encoded plot image is returned
    """
    assert len(stock) >= 3
    assert all_stocks_cip is not None
    if rules is None:
        rules = default_point_score_rules()
    rows = []
    points = 0
    day_low_high_df = day_low_high(stock, all_dates=all_stocks_cip.columns)
    state = { 'day_low_high_df': day_low_high_df,  # never changes each day, so we init it here
              'all_stocks_change_in_percent_df': all_stocks_cip,
              'stock': stock,
              'daily_range_threshold': 0.20, # 20% at either end of the daily range gets a point
            }
    for date in all_stocks_cip.columns:
        market_avg = all_stocks_cip[date].mean()
        sector_avg = all_stocks_cip[date].filter(items=sector_companies).mean()
        stock_move = all_stocks_cip.at[stock, date]
        state.update({ 'market_avg': market_avg, 'sector_avg': sector_avg,
                       'stock_move': stock_move, 'date': date })
        points += sum(map(lambda r: r(state), rules))
        rows.append({ 'points': points, 'stock': stock, 'date': date })

    df = pd.DataFrame.from_records(rows)
    df['date'] = pd.to_datetime(df['date'])
    point_score_plot = plot_series(df, x='date', y='points')
    return point_score_plot

def analyse_sector(stock, sector, all_stocks_cip, window_size=14):
    assert all_stocks_cip is not None

    sector_companies = all_sector_stocks(sector) if sector else [] # ETFs dont have a sector for now...
    if len(sector_companies) > 0:
       cip = all_stocks_cip.filter(items=sector_companies, axis='index')
       cip = cip.fillna(0.0)
       #assert len(cip) == len(sector_companies) # may fail when some stocks missing due to delisted etc.
       rows = []
       cum_sum = defaultdict(float)
       stock_versus_sector = []
       # identify the best performing stock in the sector and add it to the stock_versus_sector rows...
       best_stock_in_sector = cip.sum(axis=1).nlargest(1).index[0]
       for day in sorted(cip.columns, key=lambda k: datetime.strptime(k, "%Y-%m-%d")):
           for asx_code, daily_change in cip[day].iteritems():
               cum_sum[asx_code] += daily_change
           n_pos = len(list(filter(lambda t: t[1] >= 5.0, cum_sum.items())))
           n_neg = len(list(filter(lambda t: t[1] < -5.0, cum_sum.items())))
           n_unchanged = len(cip) - n_pos - n_neg
           rows.append({ 'n_pos': n_pos, 'n_neg': n_neg, 'n_unchanged': n_unchanged, 'date': day})
           stock_versus_sector.append({ 'group': stock, 'date': day, 'value': cum_sum[stock] })
           stock_versus_sector.append({ 'group': 'sector_average', 'date': day, 'value': pd.Series(cum_sum).mean() })
           if stock != best_stock_in_sector:
               stock_versus_sector.append({ 'group': '{} (best in {})'.format(best_stock_in_sector, sector), 'value': cum_sum[best_stock_in_sector], 'date': day})
       df = pd.DataFrame.from_records(rows)

       sector_momentum_plot = plot_sector_performance(df, sector, window_size=window_size)
       stock_versus_sector_df = pd.DataFrame.from_records(stock_versus_sector)
       c_vs_s_plot = plot_company_versus_sector(stock_versus_sector_df, stock, sector)
       point_score_plot = analyse_point_scores(stock, sector_companies, all_stocks_cip)
    else:
       c_vs_s_plot = sector_momentum_plot = point_score_plot = None

    return c_vs_s_plot, sector_momentum_plot, point_score_plot
