"""
Responsible for handling requests for pages from the website and delegating the analysis
and visualisation as required.
"""
from collections import defaultdict
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
from bson.objectid import ObjectId
from django import forms
from django.shortcuts import render
from django.core.paginator import Paginator
from django.views.generic import FormView, UpdateView, DeleteView, CreateView
from django.http import HttpResponseRedirect, Http404, HttpResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic.list import (
    MultipleObjectTemplateResponseMixin,
    MultipleObjectMixin,
)
from app.models import (
    Quotation,
    user_watchlist,
    latest_quotation_date,
    find_movers,
    Sector,
    all_available_dates,
    all_sector_stocks,
    desired_dates,
    latest_quote,
    find_named_companies,
    validate_user,
    validate_stock,
    company_prices,
    Security,
    CompanyDetails,
    all_quotes,
    all_etfs,
    increasing_eps,
    increasing_yield,
    user_purchases,
    Watchlist,
    VirtualPurchase
)
from app.mixins import SearchMixin
from app.messages import info, warning, add_messages
from app.forms import (
    SectorSearchForm,
    DividendSearchForm,
    CompanySearchForm,
    MoverSearchForm,
)
from app.analysis import (
    analyse_sector,
    calculate_trends,
    rank_cumulative_change,
    optimise_portfolio,
    detect_outliers,
    default_point_score_rules,
)
from app.plots import (
    plot_heatmap,
    make_rsi_plot,
    plot_point_scores,
    plot_key_stock_indicators,
    plot_market_wide_sector_performance,
    plot_company_rank,
    plot_portfolio,
    plot_best_monthly_price_trend
)

class DividendYieldSearch(
        SearchMixin,
        LoginRequiredMixin,
        MultipleObjectMixin,
        MultipleObjectTemplateResponseMixin,
        FormView,
):
    form_class = DividendSearchForm
    template_name = "search_form.html"  # generic template, not specific to this view
    action_url = "/search/by-yield"
    paginate_by = 50
    ordering = ("-annual_dividend_yield",)
    as_at_date = None
    n_top_bottom = 20

    def additional_context(self, context):
        """
            Return the additional fields to be added to the context by render_to_response(). Subclasses
            should override this rather than the template design pattern implementation of render_to_response()
        """
        # all() to get a fresh queryset instance
        vec = context["paginator"].object_list.all()
        # print(vec)
        qs = [stock.asx_code for stock in vec]
        if len(qs) == 0:
            warning(self.request, "No stocks to report")
            sentiment_data, df, top10, bottom10, n_stocks = (None, None, None, None, 0)
        else:
            sentiment_data, df, top10, bottom10, n_stocks = plot_heatmap(
                qs, n_top_bottom=self.n_top_bottom
            )

        return {
            "most_recent_date": self.as_at_date,
            "sentiment_heatmap": sentiment_data,
            "watched": user_watchlist(
                self.request.user
            ),  # to ensure bookmarks are correct
            "n_top_bottom": self.n_top_bottom,
            "best_ten": top10,
            "worst_ten": bottom10,
            "title": "Find by dividend yield or P/E",
            "sentiment_heatmap_title": "Recent sentiment: {} total stocks".format(
                n_stocks
            ),
        }

    def sort_by(self, qs, sort_by):
        assert qs is not None
        if sort_by is None:
            print("Sorting by {}".format(self.ordering))
            qs = qs.order_by(*self.ordering)
        else:
            sort_tuple = tuple(sort_by.split(","))
            print("Sorting by {}".format(sort_tuple))
            qs = qs.order_by(*sort_tuple)
        return qs

    def render_to_response(self, context):
        """
        A template design pattern to organise the response. Provides an additional_context()
        method for subclasses to use. Response is guaranteed to include:
        1) messages informing the user of issues
        2) additional context for the django template
        or an exception raised
        """
        context.update(self.additional_context(context))
        add_messages(self.request, context)
        return super().render_to_response(context)

    def get_queryset(self, **kwargs):
        if kwargs == {}:
            return Quotation.objects.none()

        self.as_at_date = latest_quotation_date("ANZ")
        min_yield = kwargs.get("min_yield") if "min_yield" in kwargs else 0.0
        max_yield = kwargs.get("max_yield") if "max_yield" in kwargs else 10000.0
        results = Quotation.objects.filter(fetch_date=self.as_at_date).\
                                    filter(annual_dividend_yield__gte=min_yield).\
                                    filter(annual_dividend_yield__lte=max_yield)
        
        if "min_pe" in kwargs:
            results = results.filter(pe__gte=kwargs.get("min_pe"))
        if "max_pe" in kwargs:
            results = results.filter(pe__lt=kwargs.get("max_pe"))
        if "min_eps_aud" in kwargs:
            results = results.filter(eps__gte=kwargs.get("min_eps_aud"))
        return self.sort_by(results, self.request.GET.get("sort_by"))


dividend_search = DividendYieldSearch.as_view()

class SectorSearchView(DividendYieldSearch):
    form_class = SectorSearchForm
    action_url = "/search/by-sector"
    sector = "Communication Services"   # default to Comms. Services if not specified
    sector_id = None
    query_state = None

    def additional_context(self, context):
        assert isinstance(self.sector, str)
        assert isinstance(self.sector_id, int)

        ret = {
            # to highlight top10/bottom10 bookmarks correctly
            "watched": user_watchlist(self.request.user),  
            "title": "Find by company sector",
            "sector_name": self.sector,
            "sector_id": self.sector_id,
        }
        if self.query_state is not None:
            ret.update(self.query_state)

        ret["sentiment_heatmap_title"] = "{}: past {} days".format(
            self.sector, self.query_defaults("n_days")
        )
        if 'top10' not in context:
            ret['top10'] = None
        if 'bottom10' not in context:
            ret['bottom10'] = None

        add_messages(self.request, ret)
        return ret

    def get_desired_stocks(self, wanted_stocks, want_best10, want_worst10, top10, bottom10):
        assert wanted_stocks is not None

        if want_best10 or want_worst10:
            wanted = set()
            restricted = False
            if want_best10:
                wanted = wanted.union(top10.index)
                restricted = True
            if want_worst10:
                wanted = wanted.union(bottom10.index)
                restricted = True
            if restricted:
                return set(wanted_stocks.intersection(wanted))

        return set(wanted_stocks)

    def query_defaults(self, field_name):
        if field_name == "n_days":
            return 30
        elif field_name == "n_top_bottom":
            return self.n_top_bottom
   
        assert False # field_name not known
        return None

    def get_queryset(self, **kwargs):
        # user never run this view before?
        if kwargs == {}:
            print("WARNING: no form parameters specified - returning empty queryset")
            return Quotation.objects.none()

        self.sector = kwargs.get("sector", self.sector)
        self.sector_id = int(Sector.objects.get(sector_name=self.sector).sector_id)
        wanted_stocks = all_sector_stocks(self.sector)
        print("Found {} stocks matching sector={}".format(len(wanted_stocks), self.sector))
        n_days = self.query_defaults("n_days")
        n_top_bottom = self.query_defaults("n_top_bottom")
        wanted_dates = desired_dates(start_date=n_days)
        heatmap, _, top10, bottom10, _ = plot_heatmap(
            wanted_stocks, all_dates=wanted_dates, n_top_bottom=n_top_bottom
        )

        wanted_stocks = self.get_desired_stocks(wanted_stocks, 
                                                kwargs.get('best10', False),
                                                kwargs.get('worst10', False),
                                                top10, bottom10)

        when_date = wanted_dates[-1]
        print("Looking for {} companies as at {}".format(len(wanted_stocks), when_date))
        self.query_state = {
            "most_recent_date":  when_date,
            "sentiment_heatmap": heatmap,
            "best_ten":          top10,
            "worst_ten":         bottom10,
            "wanted_stocks":     wanted_stocks,
            "n_top_bottom":      n_top_bottom,
        }
        
        results = Quotation.objects.filter(
            asx_code__in=wanted_stocks, fetch_date=when_date
        ).exclude(error_code="id-or-code-invalid")
        return self.sort_by(results, self.request.GET.get('sort_by'))

sector_search = SectorSearchView.as_view()

class MoverSearch(DividendYieldSearch):
    form_class = MoverSearchForm
    action_url = "/search/movers"

    def additional_context(self, context):
        return {"title": "Find companies with movement (%) within timeframe"}

    def get_queryset(self, **kwargs):
        if any(
            [kwargs == {}, "threshold" not in kwargs, "timeframe_in_days" not in kwargs]
        ):
            return Quotation.objects.none()
        threshold_percentage = kwargs.get("threshold")
        timeframe_in_days = kwargs.get("timeframe_in_days")
        user_sort = self.request.GET.get("sort_by")
        df = find_movers(
            threshold_percentage, desired_dates(start_date=timeframe_in_days)
        )
        matching_companies = set()
        if kwargs.get("show_increasing", False):
            matching_companies.update(df[df > 0.0].index)
        if kwargs.get("show_decreasing", False):
            matching_companies.update(df[df < 0.0].index)
        results, latest_date = latest_quote(tuple(matching_companies))
        return self.sort_by(results, user_sort)


mover_search = MoverSearch.as_view()


class CompanySearch(DividendYieldSearch):
    form_class = CompanySearchForm
    action_url = "/search/by-company"

    def additional_context(self, context):
        return {"title": "Find by company name or activity"}

    def get_queryset(self, **kwargs):
        if kwargs == {} or not any(["name" in kwargs, "activity" in kwargs]):
            return Quotation.objects.none()
        wanted_name = kwargs.get("name", "")
        wanted_activity = kwargs.get("activity", "")
        matching_companies = find_named_companies(wanted_name, wanted_activity)
        print("Showing results for {} companies".format(len(matching_companies)))
        self.as_at_date = latest_quotation_date("ANZ")
        results, _ = latest_quote(tuple(matching_companies))
        return self.sort_by(results, self.request.GET.get("sort_by"))


company_search = CompanySearch.as_view()


@login_required
def show_all_stocks(request):
    all_dates = all_available_dates()
    if len(all_dates) < 1:
        raise Http404("No ASX price data available!")
    ymd = all_dates[-1]
    assert isinstance(ymd, str) and len(ymd) > 8
    qs = (
        Quotation.objects.filter(fetch_date=ymd)
        .exclude(asx_code__isnull=True)
        .exclude(last_price__isnull=True)
        .exclude(volume=0)
        .order_by("-annual_dividend_yield", "-last_price", "-volume")
    )
    sort_by = request.GET.get("sort_by", None)
    if sort_by is not None:
        qs = qs.order_by(sort_by)
    assert qs is not None
    paginator = Paginator(qs, 50)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)
    context = {
        "title": "ASX stocks by dividend yield",
        "page_obj": page_obj,
        "most_recent_date": ymd,
        "watched": user_watchlist(request.user),
    }
    return render(request, "all_stocks.html", context=context)


@login_required
def show_stock(request, stock=None, sector_n_days=90, stock_n_days=365):
    """
    Displays a view of a single stock via the stock_view.html template and associated state
    """
    validate_stock(stock)
    validate_user(request.user)

    window_size = (
        14  # since must have a full window before computing momentum over sector_n_days
    )
    stock_dates = desired_dates(start_date=stock_n_days + window_size)
    wanted_fields = [
        "last_price",
        "volume",
        "day_low_price",
        "day_high_price",
        "eps",
        "pe",
        "annual_dividend_yield",
    ]
    stock_df = company_prices(
        [stock], all_dates=stock_dates, fields=wanted_fields, fail_missing_months=False
    )
    # print(stock_df)

    securities = Security.objects.filter(asx_code=stock)
    company_details = CompanyDetails.objects.filter(asx_code=stock).first()
    if company_details is None:
        warning(request, "No details available for {}".format(stock))

    n_dates = len(stock_df)
    if (
        n_dates < 14
    ):  # RSI requires at least 14 prices to plot so reject recently added stocks
        raise Http404(
            "Insufficient price quotes for {} - only {}".format(stock, n_dates)
        )

    # plot relative strength
    fig = make_rsi_plot(stock, stock_df)

    # show sector performance over past 3 months
    sector_dates = desired_dates(start_date=180)
    all_stocks_cip = company_prices(
        None,
        all_dates=sector_dates,
        fail_missing_months=False,
        fields="change_in_percent",
        fix_missing=False,
    )
    sector = company_details.sector_name if company_details else None
    if sector is not None:  # not an ETF? ie. sector information available?
        sector_companies = all_sector_stocks(sector)
        c_vs_s_plot, sector_momentum_plot = analyse_sector(
            stock, sector, sector_companies, all_stocks_cip, window_size=window_size
        )
        t = plot_point_scores(
            stock, sector_companies, all_stocks_cip, default_point_score_rules()
        )
        point_score_plot, net_rule_contributors_plot = t
    else:
        c_vs_s_plot = (
            sector_momentum_plot
        ) = point_score_plot = net_rule_contributors_plot = None

    # key indicator performance over past 90 days (for now): pe, eps, yield etc.
    key_indicator_plot = plot_key_stock_indicators(stock_df, stock)
    # plot the price over last 600 days in monthly blocks ie. max 24 bars which is still readable
    monthly_maximum_plot = plot_best_monthly_price_trend(
        all_quotes(stock, all_dates=desired_dates(start_date=365))
    )

    # populate template and render HTML page with context
    context = {
        "rsi_data": fig,
        "asx_code": stock,
        "securities": securities,
        "cd": company_details,
        "sector_momentum_plot": sector_momentum_plot,
        "sector_momentum_title": "{} sector stocks: {} day performance".format(
            sector, sector_n_days
        ),
        "company_versus_sector_plot": c_vs_s_plot,
        "company_versus_sector_title": "{} vs. {} performance".format(stock, sector),
        "key_indicators_plot": key_indicator_plot,
        "monthly_highest_price_plot_title": "Maximum price each month trend",
        "monthly_highest_price_plot": monthly_maximum_plot,
        "point_score_plot": point_score_plot,
        "point_score_plot_title": "Points score due to price movements",
        "net_contributors_plot": net_rule_contributors_plot,
        "net_contributors_plot_title": "Contributions to point score by rule",
        "watched": user_watchlist(request.user),
    }
    return render(request, "stock_view.html", context=context)


def save_dataframe_to_file(df, filename, output_format):
    assert output_format in ("csv", "excel", "tsv", "parquet")
    assert df is not None and len(df) > 0
    assert len(filename) > 0

    if output_format == "csv":
        df.to_csv(filename)
        return "text/csv"
    elif output_format == "excel":
        df.to_excel(filename)
        return "application/vnd.ms-excel"
    elif output_format == "tsv":
        df.to_csv(filename, sep="\t")
        return "text/tab-separated-values"
    elif output_format == "parquet":
        df.to_parquet(filename)
        return "application/octet-stream"  # for now, but must be something better...
    else:
        raise ValueError("Unsupported format {}".format(output_format))


def get_dataset(dataset_wanted):
    assert dataset_wanted in ("market_sentiment")

    if dataset_wanted == "market_sentiment":
        _, df, _, _, _ = plot_heatmap(
            None, all_dates=desired_dates(21), n_top_bottom=20
        )
        return df
    else:
        raise ValueError("Unsupported dataset {}".format(dataset_wanted))


@login_required
def download_data(request, dataset=None, output_format="csv"):
    validate_user(request.user)

    with tempfile.NamedTemporaryFile() as fh:
        df = get_dataset(dataset)
        content_type = save_dataframe_to_file(df, fh.name, output_format)
        fh.seek(0)
        response = HttpResponse(fh.read(), content_type=content_type)
        response["Content-Disposition"] = "inline; filename=temp.{}".format(output_format)
        return response


@login_required
def market_sentiment(request, n_days=21, n_top_bottom=20, sector_n_days=180):
    validate_user(request.user)
    assert n_days > 0
    assert n_top_bottom > 0
    all_dates = desired_dates(start_date=n_days)
    sentiment_heatmap_data, df, top10, bottom10, n = plot_heatmap(
        None, all_dates=all_dates, n_top_bottom=n_top_bottom
    )
    sector_performance_plot = plot_market_wide_sector_performance(
        desired_dates(start_date=sector_n_days)
    )

    context = {
        "sentiment_data": sentiment_heatmap_data,
        "n_days": n_days,
        "n_stocks_plotted": n,
        "n_top_bottom": n_top_bottom,
        "best_ten": top10,
        "worst_ten": bottom10,
        "watched": user_watchlist(request.user),
        "sector_performance": sector_performance_plot,
        "sector_performance_title": "180 day cumulative sector avg. performance",
        "title": "Market sentiment over past {} days".format(n_days),
    }
    return render(request, "market_sentiment_view.html", context=context)


@login_required
def show_etfs(request):
    validate_user(request.user)
    matching_codes = all_etfs()
    return show_matching_companies(
        matching_codes,
        "Exchange Traded funds over past 300 days",
        "Sentiment for ETFs",
        None,
        request,
    )


@login_required
def show_increasing_eps_stocks(request):
    validate_user(request.user)
    matching_companies = increasing_eps(None)
    return show_matching_companies(
        matching_companies,
        "Stocks with increasing EPS over past 300 days",
        "Sentiment for selected stocks",
        None,  # dont show purchases on this view
        request,
    )


@login_required
def show_increasing_yield_stocks(request):
    validate_user(request.user)
    matching_companies = increasing_yield(None)
    return show_matching_companies(
        matching_companies,
        "Stocks with increasing yield over past 300 days",
        "Sentiment for selected stocks",
        None,
        request,
    )


def show_outliers(request, stocks, n_days=30, extra_context=None):
    assert stocks is not None
    assert n_days is not None  # typically integer, but desired_dates() is polymorphic
    all_dates = desired_dates(start_date=n_days)
    cip = company_prices(stocks, all_dates=all_dates, fields="change_in_percent")
    outliers = detect_outliers(stocks, cip)
    return show_matching_companies(
        outliers,
        "Unusual stock behaviours over past {} days".format(n_days),
        "Outlier stocks: sentiment",
        user_purchases(request.user),
        request,
        extra_context=extra_context,
    )


def show_optimised_stocks(request, stocks, past_n_days=365, exclude=None):
    """
       Backend function to implement function-based views. Default is to use past year of data
       and perform HRP Optimisation
    """
    validate_user(request.user)

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = exclude.split(",")
        stocks = set(stocks).difference(exclude)
    (
        cleaned_weights,
        performance,
        efficient_frontier_plot,
        correlation_plot,
        messages,
    ) = optimise_portfolio(stocks, desired_dates(start_date=past_n_days))

    for msg in messages:
        warning(request, msg)

    context = {
        "n_stocks": len(stocks),
        "cleaned_weights": cleaned_weights,
        "portfolio_performance": performance,
        "efficient_frontier_plot": efficient_frontier_plot,
        "correlation_plot": correlation_plot,
    }
    add_messages(request, context)
    return render(request, "optimised_view.html", context=context)


@login_required
def show_optimised_sector(request, sector_id=None, exclude=None):
    sector_name = Sector.objects.get(sector_id=sector_id).sector_name
    stocks = all_sector_stocks(sector_name)
    return show_optimised_stocks(request, stocks, exclude=exclude)


@login_required
def show_optimised_watchlist(request, exclude=None):
    return show_optimised_stocks(request, user_watchlist(request.user), exclude=exclude)


@login_required
def show_optimised_etfs(request, exclude=None):
    return show_optimised_stocks(request, all_etfs(), exclude=exclude)


@login_required
def show_sector_outliers(request, sector_id=None, n_days=30):
    validate_user(request.user)
    assert isinstance(sector_id, int) and sector_id > 0

    stocks = all_sector_stocks(Sector.objects.get(sector_id=sector_id).sector_name)
    return show_outliers(request, stocks, n_days=n_days)


@login_required
def show_watchlist_outliers(request, n_days=30):
    validate_user(request.user)
    stocks = user_watchlist(request.user)
    return show_outliers(request, stocks, n_days=n_days)


@login_required
def show_trends(request):
    validate_user(request.user)
    watchlist_stocks = user_watchlist(request.user)
    all_dates = desired_dates(start_date=300)  # last 300 days
    cip = company_prices(
        watchlist_stocks,
        all_dates=all_dates,
        fields="change_in_percent",
        fail_missing_months=False,
    )
    trends = calculate_trends(cip, watchlist_stocks, all_dates)
    # for now we only plot trending companies... too slow and unreadable to load the page otherwise!
    cip = rank_cumulative_change(
        cip.filter(trends.keys(), axis="index"), all_dates=all_dates
    )
    trending_companies_plot = plot_company_rank(cip)
    context = {
        "watchlist_trends": trends,
        "trending_companies_plot": trending_companies_plot,
        "trending_companies_plot_title": "Trending watchlist companies by rank (past 300 days)",
    }
    return render(request, "trends.html", context=context)


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

    all_dates = desired_dates(start_date=purchase_buy_dates[0])
    df = company_prices(stocks, all_dates=all_dates)
    rows = []
    stock_count = defaultdict(int)
    stock_cost = defaultdict(float)
    portfolio_cost = 0.0

    for d in [datetime.strptime(x, "%Y-%m-%d").date() for x in all_dates]:
        d_str = str(d)
        if d_str not in df.columns:  # not a trading day?
            continue
        purchases_to_date = filter(lambda vp: vp.buy_date <= d, purchases)
        for purchase in purchases_to_date:
            if purchase.buy_date == d:
                portfolio_cost += purchase.amount
                stock_count[purchase.asx_code] += purchase.n
                stock_cost[purchase.asx_code] += purchase.amount

        portfolio_worth = sum(
            map(lambda t: df.at[t[0], d_str] * t[1], stock_count.items())
        )

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
    (
        portfolio_performance_figure,
        stock_performance_figure,
        profit_contributors_figure,
    ) = t
    context = {
        "title": "Portfolio performance",
        "portfolio_title": "Overall",
        "portfolio_figure": portfolio_performance_figure,
        "stock_title": "Stock",
        "stock_figure": stock_performance_figure,
        "profit_contributors": profit_contributors_figure,
    }
    return render(request, "portfolio_trends.html", context=context)


def show_matching_companies(
    matching_companies,
    title,
    heatmap_title,
    virtual_purchases_by_user,
    request,
    extra_context=None,
):
    """
    Support function to public-facing views to eliminate code redundancy
    """
    assert isinstance(title, str) and isinstance(heatmap_title, str)
    sort_by = request.GET.get("sort_by") or "asx_code"
    info(request, "Sorting by {}".format(sort_by))

    if len(matching_companies) > 0:
        stocks_queryset, _ = latest_quote(matching_companies)
        stocks_queryset = stocks_queryset.order_by(sort_by)
        print(
            "Found {} quotes for {} stocks".format(
                stocks_queryset.count(), len(matching_companies)
            )
        )

        # paginate results for 50 stocks per page
        paginator = Paginator(stocks_queryset, 50)
        page_number = request.GET.get("page", 1)
        page_obj = paginator.page(page_number)

        # add sentiment heatmap amongst watched stocks
        n_days = 30
        n_top_bottom = 20
        sentiment_heatmap_data, df, top10, bottom10, n = plot_heatmap(
            matching_companies,
            all_dates=desired_dates(start_date=n_days),
            n_top_bottom=n_top_bottom,
        )
    else:
        page_obj = top10 = bottom10 = sentiment_heatmap_data = None
        n_top_bottom = n_days = 0
        warning(request, "No matching companies found.")

    context = {
        "most_recent_date": latest_quotation_date("ANZ"),
        "page_obj": page_obj,
        "title": title,
        "watched": user_watchlist(request.user),
        "n_top_bottom": n_top_bottom,
        "best_ten": top10,
        "worst_ten": bottom10,
        "virtual_purchases": virtual_purchases_by_user,
        "sentiment_heatmap": sentiment_heatmap_data,
        "sentiment_heatmap_title": "{}: past {} days".format(heatmap_title, n_days),
    }
    if extra_context:
        context.update(extra_context)
    add_messages(request, context)
    return render(request, "all_stocks.html", context=context)


@login_required
def show_watched(request):
    validate_user(request.user)
    matching_companies = user_watchlist(request.user)
    purchases = user_purchases(request.user)

    return show_matching_companies(
        matching_companies,
        "Stocks you are watching",
        "Watched stock recent sentiment",
        purchases,
        request,
    )


def redirect_to_next(request, fallback_next="/"):
    """
    Call this function in your view once you have deleted some database data: set the 'next' query href
    param to where the redirect should go to. If not specified '/' will be assumed. Not permitted to
    redirect to another site.
    """
    # redirect will trigger a redraw which will show the purchase since next will be the same page
    assert request is not None
    if request.GET is not None:
        next_href = request.GET.get("next", fallback_next)
        assert next_href.startswith("/")  # PARANOIA: must be same origin
        return HttpResponseRedirect(next_href)
    else:
        return HttpResponseRedirect(fallback_next)


@login_required
def toggle_watched(request, stock=None):
    validate_stock(stock)
    validate_user(request.user)
    current_watchlist = user_watchlist(request.user)
    if stock in current_watchlist:  # remove from watchlist?
        Watchlist.objects.filter(user=request.user, asx_code=stock).delete()
    else:
        Watchlist(user=request.user, asx_code=stock).save()

    return redirect_to_next(request)


class BuyVirtualStock(LoginRequiredMixin, CreateView):
    model = VirtualPurchase
    success_url = "/show/watched"
    form_class = forms.models.modelform_factory(
        VirtualPurchase,
        fields=["asx_code", "buy_date", "price_at_buy_date", "amount", "n"],
        widgets={"asx_code": forms.TextInput(attrs={"readonly": "readonly"})},
    )

    def form_valid(self, form):
        req = self.request
        resp = super().form_valid(form)  # only if no exception raised do we save...
        self.object = form.save(commit=False)
        self.object.user = validate_user(req.user)
        self.object.save()
        info(req, "Saved purchase of {}".format(self.kwargs.get("stock")))
        return resp

    def get_context_data(self, **kwargs):
        result = super().get_context_data(**kwargs)
        result["title"] = "Add {} purchase to watchlist".format(
            self.kwargs.get("stock")
        )
        return result

    def get_initial(self, **kwargs):
        stock = self.kwargs.get("stock")
        amount = self.kwargs.get("amount", 5000.0)
        user = self.request.user
        validate_stock(stock)
        validate_user(user)
        quote, latest_date = latest_quote(stock)
        cur_price = quote.last_price
        if cur_price >= 1e-6:
            return {
                "asx_code": stock,
                "user": user,
                "buy_date": latest_date,
                "price_at_buy_date": cur_price,
                "amount": amount,
                "n": int(amount / cur_price),
            }
        else:
            warning(
                self.request, "Cannot buy {} as its price is zero/unknown".format(stock)
            )
            return {}


buy_virtual_stock = BuyVirtualStock.as_view()


class MyObjectMixin:
    """
    Retrieve the object by mongo _id for use by CRUD CBV views for VirtualPurchase's
    """

    def get_object(self, queryset=None):
        slug = self.kwargs.get("slug")
        purchase = VirtualPurchase.objects.mongo_find_one({"_id": ObjectId(slug)})
        # print(purchase)
        purchase["id"] = purchase["_id"]
        purchase.pop("_id", None)
        return VirtualPurchase(**purchase)


class EditVirtualStock(LoginRequiredMixin, MyObjectMixin, UpdateView):
    model = VirtualPurchase
    success_url = "/show/watched"
    form_class = forms.models.modelform_factory(
        VirtualPurchase,
        fields=["asx_code", "buy_date", "price_at_buy_date", "amount", "n"],
        widgets={
            "asx_code": forms.TextInput(attrs={"readonly": "readonly"}),
            "buy_date": forms.DateInput(),
        },
    )


edit_virtual_stock = EditVirtualStock.as_view()


class DeleteVirtualPurchaseView(LoginRequiredMixin, MyObjectMixin, DeleteView):
    model = VirtualPurchase
    success_url = "/show/watched"


delete_virtual_stock = DeleteVirtualPurchaseView.as_view()
