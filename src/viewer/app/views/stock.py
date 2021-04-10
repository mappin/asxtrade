from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from app.models import (validate_stock, validate_user, stock_info, cached_all_stocks_cip, 
                        Timeframe, company_prices, rsi_data, user_watchlist, selected_cached_stocks_cip, 
                        validate_date, user_purchases, all_available_dates)
from app.analysis import analyse_sector_performance, default_point_score_rules, rank_cumulative_change, calculate_trends
from app.messages import warning
from app.plots import plot_point_scores, plot_fundamentals, make_rsi_plot, plot_trend, plot_company_rank, plot_portfolio

@login_required
def show_stock_sector(request, stock):
    validate_stock(stock)
    validate_user(request.user)

    _, company_details = stock_info(stock, lambda msg: warning(request, msg))
    sector = company_details.sector_name if company_details else None
    all_stocks_cip = cached_all_stocks_cip(Timeframe(past_n_days=180))

    # invoke separate function to cache the calls when we can
    c_vs_s_plot, sector_momentum_plot, sector_companies = analyse_sector_performance(stock, sector, all_stocks_cip)
    point_score_plot = net_rule_contributors_plot = None
    if sector_companies is not None:
        point_score_plot, net_rule_contributors_plot = \
                plot_point_scores(stock,
                                  sector_companies,
                                  all_stocks_cip,
                                  default_point_score_rules())
                        
    context = {
        "is_sector": True,
        "asx_code": stock,
        "sector_momentum_plot": sector_momentum_plot,
        "sector_momentum_title": "{} sector stocks".format(sector),
        "company_versus_sector_plot": c_vs_s_plot,
        "company_versus_sector_title": "{} vs. {} performance".format(stock, sector),
        "point_score_plot": point_score_plot,
        "point_score_plot_title": "Points score due to price movements",
        "net_contributors_plot": net_rule_contributors_plot,
        "net_contributors_plot_title": "Contributions to point score by rule",
    }
    return render(request, "stock_sector.html", context)

@login_required
def show_fundamentals(request, stock=None, n_days=2 * 365):
    validate_user(request.user)
    validate_stock(stock)
    timeframe = Timeframe(past_n_days=n_days)
    df = company_prices(
        [stock],
        timeframe,
        fields=("eps", "volume", "last_price", "annual_dividend_yield", \
                "pe", "change_in_percent", "change_price", "market_cap", \
                "number_of_shares"),
        missing_cb=None
    )
    #print(df)
    df['change_in_percent_cumulative'] = df['change_in_percent'].cumsum() # nicer to display cumulative
    df = df.drop('change_in_percent', axis=1)
    fundamentals_plot = plot_fundamentals(df, stock)
    context = {
        "asx_code": stock,
        "is_fundamentals": True,
        "fundamentals_plot": fundamentals_plot
    }
    return render(request, "stock_fundamentals.html", context)

@login_required
def show_stock(request, stock=None, n_days=2 * 365):
    """
    Displays a view of a single stock via the stock_view.html template and associated state
    """
    validate_stock(stock)
    validate_user(request.user)

    timeframe = Timeframe(past_n_days=n_days+200) # add 200 days so MA 200 can initialise itself before the plotting starts...
    stock_df = rsi_data(stock, timeframe) # may raise 404 if too little data available
    securities, company_details = stock_info(stock, lambda msg: warning(request, msg))

    momentum_plot = make_rsi_plot(stock, stock_df)

    # plot the price over timeframe in monthly blocks
    prices = stock_df[['last_price']].transpose() # use list of columns to ensure pd.DataFrame not pd.Series
    #print(prices)
    monthly_maximum_plot = plot_trend(prices, sample_period='M')

    # populate template and render HTML page with context
    context = {
        "asx_code": stock,
        "securities": securities,
        "cd": company_details,
        "rsi_plot": momentum_plot,
        "is_momentum": True,
        "monthly_highest_price_plot_title": "Maximum price each month trend",
        "monthly_highest_price_plot": monthly_maximum_plot,
        "timeframe": f"{n_days} days",
        "watched": user_watchlist(request.user),
    }
    return render(request, "stock_view.html", context=context)


@login_required
def show_trends(request):
    validate_user(request.user)
    watchlist_stocks = user_watchlist(request.user)
    timeframe = Timeframe(past_n_days=300)
    cip = selected_cached_stocks_cip(watchlist_stocks, timeframe)
    trends = calculate_trends(cip, watchlist_stocks)
    #print(trends)
    # for now we only plot trending companies... too slow and unreadable to load the page otherwise!
    cip = rank_cumulative_change(
        cip.filter(trends.keys(), axis="index"), timeframe
    )
    #print(cip)
    trending_companies_plot = plot_company_rank(cip)
    context = {
        "watchlist_trends": trends,
        "trending_companies_plot": trending_companies_plot,
        "trending_companies_plot_title": "Trending watchlist companies by rank: {}".format(timeframe.description),
    }
    return render(request, "trends.html", context=context)


def sum_portfolio(df: pd.DataFrame, date_str: str, stock_items):
    validate_date(date_str)

    portfolio_worth = sum(map(lambda t: df.at[t[0], date_str] * t[1], stock_items))
    return portfolio_worth

@login_required
def show_purchase_performance(request):
    purchase_buy_dates = []
    purchases = []
    stocks = []
    for stock, purchases_for_stock in user_purchases(request.user).items():
        stocks.append(stock)
        for purchase in purchases_for_stock:
            purchase_buy_dates.append(purchase.buy_date)
            purchases.append(purchase)

    purchase_buy_dates = sorted(purchase_buy_dates)
    # print("earliest {} latest {}".format(purchase_buy_dates[0], purchase_buy_dates[-1]))

    timeframe = Timeframe(from_date=str(purchase_buy_dates[0]), to_date=all_available_dates()[-1])
    df = company_prices(stocks, timeframe, transpose=True)
    rows = []
    stock_count = defaultdict(int)
    stock_cost = defaultdict(float)
    portfolio_cost = 0.0

    for d in [datetime.strptime(x, "%Y-%m-%d").date() for x in timeframe.all_dates()]:
        d_str = str(d)
        if d_str not in df.columns:  # not a trading day?
            continue
        purchases_to_date = filter(lambda vp, d=d: vp.buy_date <= d, purchases)
        for purchase in purchases_to_date:
            if purchase.buy_date == d:
                portfolio_cost += purchase.amount
                stock_count[purchase.asx_code] += purchase.n
                stock_cost[purchase.asx_code] += purchase.amount

        portfolio_worth = sum_portfolio(df, d_str, stock_count.items())
        #print(df)
        # emit rows for each stock and aggregate portfolio
        for asx_code in stocks:
            cur_price = df.at[asx_code, d_str]
            if np.isnan(cur_price):  # price missing? ok, skip record
                continue
            assert cur_price is not None and cur_price >= 0.0
            stock_worth = cur_price * stock_count[asx_code]

            rows.append(
                {
                    "portfolio_cost": portfolio_cost,
                    "portfolio_worth": portfolio_worth,
                    "portfolio_profit": portfolio_worth - portfolio_cost,
                    "stock_cost": stock_cost[asx_code],
                    "stock_worth": stock_worth,
                    "stock_profit": stock_worth - stock_cost[asx_code],
                    "date": d_str,
                    "stock": asx_code,
                }
            )

    t = plot_portfolio(pd.DataFrame.from_records(rows))
    portfolio_performance_figure, stock_performance_figure, profit_contributors_figure = t
    context = {
        "title": "Portfolio performance",
        "portfolio_title": "Overall",
        "portfolio_figure": portfolio_performance_figure,
        "stock_title": "Stock",
        "stock_figure": stock_performance_figure,
        "profit_contributors": profit_contributors_figure,
    }
    return render(request, "portfolio_trends.html", context=context)
