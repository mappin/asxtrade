"""
Responsible for handling requests for pages from the website and delegating the analysis
and visualisation as required.
"""
from numpy import isnan
import pandas as pd
from collections import defaultdict
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from app.models import (
    user_watchlist,
    latest_quotation_date,
    cached_all_stocks_cip,
    company_prices,
    Timeframe,
    validate_user,
    stocks_by_sector
)
from app.plots import (
    plot_heatmap,
    plot_market_wide_sector_performance,
    plot_market_cap_distribution,
    plot_sector_field,
)

@login_required
def market_sentiment(request, n_days=21, n_top_bottom=20, sector_n_days=180):
    validate_user(request.user)
    assert n_days > 0
    assert n_top_bottom > 0
    timeframe = Timeframe(past_n_days=n_days)
    sector_timeframe = Timeframe(past_n_days=sector_n_days)
    df = cached_all_stocks_cip(timeframe)
    sector_df = cached_all_stocks_cip(sector_timeframe)
    sentiment_plot, top10, bottom10 = plot_heatmap(df, timeframe, n_top_bottom=n_top_bottom)
    sector_performance_plot = plot_market_wide_sector_performance(sector_df)

    context = {
        "sentiment_data": sentiment_plot,
        "n_days": timeframe.n_days,
        "n_stocks_plotted": len(df),
        "n_top_bottom": n_top_bottom,
        "best_ten": top10,
        "worst_ten": bottom10,
        "watched": user_watchlist(request.user),
        "sector_performance": sector_performance_plot,
        "sector_performance_title": "Cumulative sector avg. performance: {}".format(sector_timeframe.description),
        "title": "Market sentiment: {}".format(timeframe.description),
        "market_cap_distribution_plot": plot_market_cap_distribution(tuple(df.index), latest_quotation_date('ANZ'), sector_df.columns[0])
    }
    return render(request, "market_sentiment_view.html", context=context)


@login_required
def show_pe_trends(request):
    """
    Display a plot of per-sector PE trends across stocks in each sector
    """
    validate_user(request.user)
    timeframe = Timeframe(past_n_days=180)
    pe = company_prices(None, timeframe, fields="pe", transpose=True)
    eps_df = company_prices(None, timeframe, fields="eps", transpose=True)
    ss = stocks_by_sector()
    df = pe.merge(ss, left_index=True, right_on='asx_code')
    df = df.set_index('asx_code')
    n_stocks = len(df)
    exclude_zero_sum = df[df.sum(axis=1) > 0.0]
    n_non_zero_sum = len(exclude_zero_sum)
    #print(exclude_zero_sum)
    records = []
    trading_dates = set(exclude_zero_sum.columns)
    for ymd in filter(lambda d: d in trading_dates, timeframe.all_dates()):  # needed to avoid KeyError raised during DataFrame.at[] calls below
        n_companies_per_sector = defaultdict(int)
        sum_pe_per_sector = defaultdict(float)
        sum_eps_per_sector = defaultdict(float)
        for stock in exclude_zero_sum.index:
            pe = exclude_zero_sum.at[stock, ymd]
            eps = eps_df.at[stock, ymd]
            sector = exclude_zero_sum.at[stock, 'sector_name']
            if isnan(pe):
                continue
            #print(pe)
            assert pe >= 0.0
            assert isinstance(sector, str)
            sum_pe_per_sector[sector] += pe
            n_companies_per_sector[sector] += 1
            sum_eps_per_sector[sector] += eps

        for sum_tuple, n_tuple, eps_tuple in zip(sum_pe_per_sector.items(), n_companies_per_sector.items(), sum_eps_per_sector.items()):
            sum_sector, sum_pe = sum_tuple
            n_sector, n = n_tuple
            eps_sector, sum_eps = eps_tuple
            assert n_sector == sum_sector
            assert eps_sector == sum_sector
            assert n > 0
            assert sum_pe >= 0.0
            records.append({'sector': n_sector, 'date': ymd, 'mean_pe': sum_pe / n, 'n_stocks': n, 'sum_eps': sum_eps, 'mean_eps': sum_eps / n})
    df = pd.DataFrame.from_records(records)

    #print(df)
    context = {
        "title": "PE Trends: {}".format(timeframe.description),
        "n_stocks": n_stocks,
        "timeframe": timeframe,
        "n_stocks_with_pe": n_non_zero_sum,
        "sector_pe_plot": plot_sector_field(df, field="mean_pe"),
        "sector_eps_plot": plot_sector_field(df, field="sum_eps")
    }
    return render(request, "pe_trends.html", context)