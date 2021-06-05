"""
Responsible for providing detiled views about a single stock and closely related views
"""
from matplotlib.pyplot import subplots_adjust
import pandas as pd
import plotnine as p9
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import Http404
from app.models import (
    stocks_by_sector,
    validate_stock,
    validate_user,
    stock_info,
    cached_all_stocks_cip,
    companies_with_same_sector,
    Timeframe,
    company_prices,
    rsi_data,
    user_watchlist,
    selected_cached_stocks_cip,
    timing,
    financial_metrics,
)
from app.analysis import (
    default_point_score_rules,
    rank_cumulative_change,
    calculate_trends,
)
from app.messages import warning
from app.data import cache_plot, make_point_score_dataframe, label_shorten, pe_trends_df
from app.plots import (
    user_theme,
    plot_point_scores,
    plot_points_by_rule,
    plot_fundamentals,
    plot_momentum,
    plot_trend,
    plot_price_trend,
    plot_company_rank,
    cached_portfolio_performance,
    cached_sector_performance,
    cached_company_versus_sector,
)
from plotnine.guides.guide_colorbar import guide_colorbar


@timing
def make_stock_sector(timeframe: Timeframe, stock: str) -> dict:
    cip = cached_all_stocks_cip(timeframe)
    sector_companies = companies_with_same_sector(stock)
    sector = stock_info(stock).get("sector_name", "")

    # implement caching (in memory) at image level to avoid all data manipulation if at all possible
    sector_momentum_plot = cached_sector_performance(sector, sector_companies, cip)
    c_vs_s_plot = cached_company_versus_sector(stock, sector, sector_companies, cip)

    # invoke separate function to cache the calls when we can
    point_score_plot = net_rule_contributors_plot = None
    if len(sector_companies) > 0:
        df, net_points_by_rule = make_point_score_dataframe(
            stock, sector_companies, cip, default_point_score_rules()
        )
        ps_cache_key = f"{timeframe.description}-{stock}-point-score-plot"
        np_cache_key = f"{timeframe.description}-{stock}-rules-by-points"
        point_score_plot = plot_point_scores(ps_cache_key, df)
        net_rule_contributors_plot = plot_points_by_rule(
            np_cache_key, net_points_by_rule
        )

    return {
        "timeframe": timeframe,
        "sector_momentum": {
            "plot_uri": sector_momentum_plot,
            "title": "{} sector stocks".format(sector),
        },
        "company_versus_sector": {
            "plot_uri": c_vs_s_plot,
            "title": "Performance against sector",
        },
        "point_score": {
            "plot_uri": point_score_plot,
            "title": "Points score due to price movements",
        },
        "net_contributors": {
            "plot_uri": net_rule_contributors_plot,
            "title": "Contributions to point score by rule",
        },
    }


def make_fundamentals(timeframe: Timeframe, stock: str) -> dict:
    """Return a dict of the fundamentals plots for the current django template render to use"""

    def inner():
        df = company_prices(
            [stock],
            timeframe,
            fields=(
                "eps",
                "volume",
                "last_price",
                "annual_dividend_yield",
                "pe",
                "change_in_percent",
                "change_price",
                "market_cap",
                "number_of_shares",
            ),
            missing_cb=None,
        )
        # print(df)
        df["change_in_percent_cumulative"] = df[
            "change_in_percent"
        ].cumsum()  # nicer to display cumulative
        df = df.drop("change_in_percent", axis=1)
        df["volume"] = (
            df["last_price"] * df["volume"] / 1000000
        )  # again, express as $(M)
        df["market_cap"] /= 1000 * 1000
        df["number_of_shares"] /= 1000 * 1000
        df["fetch_date"] = pd.to_datetime(df.index, format="%Y-%m-%d")
        # print(df.shape)
        df = df.set_index("fetch_date")
        df = df.resample(
            "B"
        ).asfreq()  # fill gaps in dataframe with business day dates only
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(df)
        df["fetch_date"] = pd.to_datetime(df.index, format="%Y-%m-%d")
        # print(df.shape)
        return plot_fundamentals(df, stock)

    return {
        "plot_uri": cache_plot(
            f"{stock}-{timeframe.description}-fundamentals-plot", inner
        ),
        "title": "Stock fundamentals: EPS, PE, DY etc.",
        "timeframe": timeframe,
    }


@login_required
def show_financial_metrics(request, stock=None):
    validate_user(request.user)
    validate_stock(stock)
    data_df = financial_metrics(stock)
    if data_df is None or len(data_df) < 1:
        raise Http404(f"No financial metrics available for {stock}")

    linear_metrics = calculate_trends(data_df)
    good_linear_metrics = []
    for k, t in linear_metrics.items():
        if t[1] < 0.1:
            good_linear_metrics.append(k)
    exp_metrics = calculate_trends(data_df, polynomial_degree=3, nrmse_cutoff=0.1)
    good_exp_metrics = []
    for k, t in exp_metrics.items():
        if t[1] < 0.1:
            good_exp_metrics.append(k)
    print(
        f"n_metrics == {len(data_df)} n_trending={len(linear_metrics.keys())} n_good_fit={len(good_linear_metrics)} n_good_exp={len(good_exp_metrics)}"
    )

    def plot_metrics(df: pd.DataFrame):
        plot = (
            p9.ggplot(df, p9.aes(x="date", y="value", colour="metric"))
            + p9.geom_line(size=1.3)
            + p9.geom_point(size=3)
            #  + p9.scale_y_continuous(labels=label_shorten)
        )
        n_metrics = df["metric"].nunique()
        return user_theme(
            plot,
            subplots_adjust={"left": 0.2},
            figure_size=(12, int(n_metrics * 1.5)),
        )

    def linear_trending_metrics():
        df = data_df.filter(good_linear_metrics, axis=0)
        if len(df) < 1:
            return None
        df["metric"] = df.index
        df = df.melt(id_vars="metric").dropna(how="any", axis=0)
        plot = plot_metrics(df)
        plot += p9.facet_wrap("~metric", ncol=1, scales="free_y")
        return plot

    def exponential_growth_metrics():
        df = data_df.filter(good_exp_metrics, axis=0)
        if len(df) < 1:
            return None
        df["metric"] = df.index
        df = df.melt(id_vars="metric").dropna(how="any", axis=0)
        plot = plot_metrics(df)
        plot += p9.facet_wrap("~metric", ncol=1, scales="free_y")

        return plot

    def inner():
        df = data_df.filter(["Ebit", "Total Revenue", "Earnings"], axis=0)
        if len(df) < 2:
            print(f"WARNING: revenue and earnings not availabe for {stock}")
            return None
        df["metric"] = df.index
        df = df.melt(id_vars="metric").dropna(how="any", axis=0)
        plot = plot_metrics(df)
        return plot

    er_uri = cache_plot(f"{stock}-earnings-revenue-plot", inner)
    trending_metrics_uri = cache_plot(
        f"{stock}-trending-metrics-plot", linear_trending_metrics
    )
    exp_growth_metrics_uri = cache_plot(
        f"{stock}-exponential-growth-metrics-plot", exponential_growth_metrics
    )
    warning(
        request,
        "Due to experimental data ingest - data on this page may be wrong/misleading/inaccurate/missing. Use at own risk.",
    )
    context = {
        "asx_code": stock,
        "data": data_df,
        "earnings_and_revenue_plot_uri": er_uri,
        "trending_metrics_plot_uri": trending_metrics_uri,
        "exp_growth_metrics_plot_uri": exp_growth_metrics_uri,
    }
    return render(request, "stock_financial_metrics.html", context=context)


@login_required
def show_stock(request, stock=None, n_days=2 * 365):
    """
    Displays a view of a single stock via the template and associated state
    """
    validate_stock(stock)
    validate_user(request.user)
    plot_timeframe = Timeframe(past_n_days=n_days)  # for template

    @timing
    def data_factory():
        timeframe = Timeframe(
            past_n_days=n_days + 200
        )  # add 200 days so MA 200 can initialise itself before the plotting starts...
        stock_df = rsi_data(
            stock, timeframe
        )  # may raise 404 if too little data available
        prices = stock_df[
            ["last_price"]
        ].transpose()  # use list of columns to ensure pd.DataFrame not pd.Series
        prices = prices.filter(
            items=plot_timeframe.all_dates(), axis="columns"
        )  # drop any date in "warm up" period
        return stock_df, prices

    # key dynamic images and text for HTML response. We only compute the required data if image(s) not cached
    company_details = stock_info(stock, lambda msg: warning(request, msg))
    momentum_plot = cache_plot(
        f"{plot_timeframe.description}-{stock}-rsi-plot",
        lambda: plot_momentum(data_factory, stock, plot_timeframe.earliest_date),
        dont_cache=True,
    )
    monthly_maximum_plot = cache_plot(
        f"{plot_timeframe.description}-{stock}-monthly-maximum-plot",
        lambda: plot_trend(data_factory, sample_period="M"),
    )

    # populate template and render HTML page with context
    context = {
        "asx_code": stock,
        "watched": user_watchlist(request.user),
        "timeframe": plot_timeframe,
        "information": company_details,
        "momentum": {
            "rsi_plot": momentum_plot,
            "monthly_highest_price": {
                "title": "Highest price each month",
                "plot_uri": monthly_maximum_plot,
            },
        },
        "fundamentals": make_fundamentals(plot_timeframe, stock),
        "stock_vs_sector": make_stock_sector(plot_timeframe, stock),
    }
    return render(request, "stock_page.html", context=context)


@login_required
def show_trends(request):
    user = request.user
    validate_user(user)
    stocks = user_watchlist(user)
    timeframe = Timeframe(past_n_days=300)
    trends = None  # initialised by data_factory() and returned IFF data_factory is called during cache_plot()

    def data_factory():
        cip = selected_cached_stocks_cip(stocks, timeframe)
        trends = calculate_trends(cip)
        # print(trends)
        # for now we only plot trending companies... too slow and unreadable to load the page otherwise!
        cip = rank_cumulative_change(cip.filter(trends.keys(), axis="index"), timeframe)
        # print(cip)
        return cip, trends

    trending_companies_plot = cache_plot(
        f"{user.username}-watchlist-trends", lambda: plot_company_rank(data_factory)
    )
    # if trends is None (ie. image cached) then we must compute it for the response
    if trends is None:
        _, trends = data_factory()

    context = {
        "watchlist_trends": trends,
        "timeframe": timeframe,
        "trending_companies_uri": trending_companies_plot,
        "trending_companies_plot_title": "Trending watchlist stocks (ranked): {}".format(
            timeframe.description
        ),
    }
    return render(request, "watchlist-rank.html", context=context)


@login_required
def show_purchase_performance(request):
    validate_user(request.user)

    (
        portfolio_performance_uri,
        stock_performance_uri,
        contributors_uri,
    ) = cached_portfolio_performance(request.user)

    context = {
        "title": "Portfolio performance",
        "portfolio_title": "Overall",
        "performance_uri": portfolio_performance_uri,
        "stock_title": "Stock",
        "stock_performance_uri": stock_performance_uri,
        "contributors_uri": contributors_uri,
    }
    return render(request, "portfolio_trends.html", context=context)


@login_required
def show_total_earnings(request):
    validate_user(request.user)
    timeframe = Timeframe(past_n_days=180)

    def data_factory(timeframe: Timeframe) -> pd.DataFrame:
        df, n_stocks = pe_trends_df(timeframe)
        df = df.pivot(
            index=["asx_code", "fetch_date"], columns="field_name", values="field_value"
        )
        df = df[df["number_of_shares"] > 0]  # ignore stocks which have unknown shares
        # print(df)
        df["total_earnings"] = df["eps"] * df["number_of_shares"]
        df = df.dropna(how="any", axis=0)
        # df = df[df["total_earnings"] > 0]  # ignore stocks burning cash
        df = df.reset_index()
        df = df.pivot(index="asx_code", columns="fetch_date", values="total_earnings")
        df = df.merge(stocks_by_sector(), left_index=True, right_on="asx_code")
        df = df.set_index("asx_code", drop=True)
        df = df.groupby("sector_name").sum()
        df["sector_name"] = df.index
        df = df.melt(id_vars="sector_name", var_name="fetch_date")
        assert set(df.columns) == set(["sector_name", "fetch_date", "value"])
        df["fetch_date"] = pd.to_datetime(df["fetch_date"])
        return df

    def plot(df: pd.DataFrame) -> p9.ggplot:
        plot = (
            p9.ggplot(
                df,
                p9.aes(
                    x="fetch_date", y="value", color="sector_name", group="sector_name"
                ),
            )
            + p9.geom_line(size=1.2)
            + p9.facet_wrap("~sector_name", ncol=2, scales="free_y")
            + p9.scale_y_continuous(labels=label_shorten)
        )
        return user_theme(
            plot,
            y_axis_label="Total sector earnings ($AUD, positive contributions only)",
            figure_size=(12, 14),
            subplots_adjust={"wspace": 0.25},
        )

    context = {
        "title": "Earnings per sector over time",
        "timeframe": timeframe,
        "plot_uri": cache_plot(
            f"total-earnings-by-sector:{timeframe.description}",
            lambda: plot(data_factory(timeframe)),
        ),
    }
    return render(request, "total_earnings_by_sector.html", context=context)