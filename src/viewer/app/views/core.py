"""
Responsible for the core view - showing lists (paginated) of stocks handling all required corner conditions for missing data.
Other views are derived from this (both FBV and CBV) so be careful to take sub-views needs into account here.
"""
from django.db.models.query import QuerySet
from django.core.paginator import Paginator
from django.core.cache import cache
from django.shortcuts import render
from django.http import Http404, HttpResponse
from django.contrib.auth.decorators import login_required
from typing import Callable, Iterable
import plotnine as p9
import pandas as pd
import numpy as np
from app.models import (
    Timeframe,
    timing,
    stocks_by_sector,
    user_purchases,
    latest_quote,
    user_watchlist,
    selected_cached_stocks_cip,
    valid_quotes_only,
    all_available_dates,
    validate_date,
    validate_user,
    all_etfs,
    increasing_yield,
    increasing_eps,
)
from app.messages import info, warning, add_messages
from app.plots import cached_heatmap, plot_breakdown, user_theme
from app.data import cache_plot


@login_required
def png(request, key):
    """
    The cache contains all users of the site. In theory if key is guessed, then users can access each others content. For this reason
    all content is image only (not data) and the key is always a SHA256 hash with random content making it extremely difficult to guess.
    As further security PNG images served from the cache can only be accessed with a valid login.
    """
    assert (
        request is not None
    )  # dont yet require user login for this endpoint, but maybe we should???
    try:
        byte_value = cache.get(key)
        assert isinstance(byte_value, bytes)
        response = HttpResponse(byte_value, content_type="image/png")
        # response['X-Accel-Redirect'] = reader.name
        return response
    except KeyError:
        raise Http404("No such cached image: {}".format(key))


def show_companies(
    matching_companies,  # may be QuerySet or iterable of stock codes (str)
    request,
    sentiment_timeframe: Timeframe,
    extra_context=None,
    template_name="all_stocks.html",
):
    """
    Support function to public-facing views to eliminate code redundancy
    """
    virtual_purchases_by_user = user_purchases(request.user)

    if isinstance(matching_companies, QuerySet):
        stocks_queryset = matching_companies  # we assume QuerySet is already sorted by desired criteria
    elif matching_companies is None or len(matching_companies) > 0:
        stocks_queryset, _ = latest_quote(matching_companies)
        # FALLTHRU

    # sort queryset as this will often be requested by the USER
    arg = request.GET.get("sort_by", "asx_code")
    info(request, "Sorting by {}".format(arg))

    if arg == "sector" or arg == "sector,-eps":
        ss = {
            s["asx_code"]: s["sector_name"]
            for s in stocks_by_sector().to_dict("records")
        }
        if arg == "sector":
            stocks_queryset = sorted(
                stocks_queryset, key=lambda s: ss.get(s.asx_code, "Z")
            )  # companies without sector sort last
        else:
            stocks_queryset = sorted(
                stocks_queryset, key=lambda s: (ss.get(s.asx_code, "Z"), -s.eps)
            )
    else:
        sort_by = tuple(arg.split(","))
        stocks_queryset = stocks_queryset.order_by(*sort_by)

    # keep track of stock codes for template convenience
    asx_codes = [quote.asx_code for quote in stocks_queryset]
    n_top_bottom = (
        extra_context["n_top_bottom"] if "n_top_bottom" in extra_context else 20
    )
    print("show_companies: found {} stocks".format(len(asx_codes)))

    # setup context dict for the render
    context = {
        # NB: title and heatmap_title are expected to be supplied by caller via extra_context
        "timeframe": sentiment_timeframe,
        "title": "Caller must override",
        "watched": user_watchlist(request.user),
        "n_stocks": len(asx_codes),
        "n_top_bottom": n_top_bottom,
        "virtual_purchases": virtual_purchases_by_user,
    }

    # since we sort above, we must setup the pagination also...
    # assert isinstance(stocks_queryset, QuerySet)
    paginator = Paginator(stocks_queryset, 50)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.page(page_number)
    context["page_obj"] = page_obj
    context["object_list"] = paginator

    @timing
    def data_factory() -> tuple:
        # compute totals across all dates for the specified companies to look at top10/bottom10 in the timeframe
        cip = selected_cached_stocks_cip(asx_codes, sentiment_timeframe)
        sum_by_company = cip.sum(axis=1)
        top10 = sum_by_company.nlargest(n_top_bottom)
        bottom10 = sum_by_company.nsmallest(n_top_bottom)

        return cip, top10, bottom10

    if len(asx_codes) <= 0:
        warning(request, "No matching companies found.")
    else:
        sentiment_heatmap_uri = cached_heatmap(
            asx_codes,
            sentiment_timeframe,
            lambda: selected_cached_stocks_cip(asx_codes, sentiment_timeframe),
        )
        key = (
            "-".join(asx_codes)
            + "-"
            + sentiment_timeframe.description
            + "-breakdown-plot"
        )
        (breakdown_plot, top10, bottom10) = plot_breakdown(data_factory)
        sector_breakdown_uri = cache_plot(key, lambda: breakdown_plot)

        def plot_cumulative_returns(
            data_factory: Callable, wanted_stocks: Iterable[str]
        ) -> p9.ggplot:
            # print(wanted_stocks)
            df, _, _ = data_factory()
            df = df.filter(wanted_stocks, axis=0).cumsum(axis=1)
            dates = df.columns
            df["asx_code"] = df.index
            df = df.melt(value_vars=dates, id_vars="asx_code")
            df["fetch_date"] = pd.to_datetime(df["fetch_date"], format="%Y-%m-%d")
            # smooth each line to make the plot more readable
            textual_df = df[df["fetch_date"] == dates[-1]]
            df["rank"] = pd.qcut(df["value"], 10, labels=False, duplicates="drop")
            df["rank"] = np.clip((df["rank"] / 10) + 0.1, 0.4, 1.0)
            df["rank"] = df["rank"].fillna(value=0.4)

            plot = (
                p9.ggplot(
                    df,
                    p9.aes(
                        "fetch_date",
                        "value",
                        group="asx_code",
                        colour="asx_code",
                    ),
                )
                + p9.geom_line(p9.aes(alpha="rank"), size=1.3)
                + p9.geom_text(
                    textual_df,
                    p9.aes(x="fetch_date", y="value", label="asx_code"),
                    color="black",
                    size=9,
                    position=p9.position_jitter(width=1, height=1),
                )
            )

            return user_theme(plot, y_axis_label="Cumulative Return (%)")

        top10_plot_uri = cache_plot(
            f"top10-plot-{'-'.join(top10.index)}",
            lambda: plot_cumulative_returns(data_factory, top10.index),
        )
        bottom10_plot_uri = cache_plot(
            f"bottom10-plot-{'-'.join(bottom10.index)}",
            lambda: plot_cumulative_returns(data_factory, bottom10.index),
        )

        context.update(
            {
                "best_ten": top10,
                "worst_ten": bottom10,
                "sentiment_heatmap_uri": sentiment_heatmap_uri,
                "sentiment_heatmap_title": "{}: {}".format(
                    context["title"], sentiment_timeframe.description
                ),
                "sector_breakdown_uri": sector_breakdown_uri,
                "top10_plot_uri": top10_plot_uri,
                "bottom10_plot_uri": bottom10_plot_uri,
            }
        )

    if extra_context:
        context.update(extra_context)
    add_messages(request, context)
    # print(context)
    return render(request, template_name, context=context)


@login_required
def show_all_stocks(request):
    all_dates = all_available_dates()
    if len(all_dates) < 1:
        raise Http404("No ASX price data available!")
    ymd = all_dates[-1]
    validate_date(ymd)
    qs, _ = valid_quotes_only(ymd)
    timeframe = Timeframe()
    return show_companies(
        qs,
        request,
        timeframe,
        extra_context={
            "title": "All stocks",
            "sentiment_heatmap_title": "All stock sentiment: {}".format(
                timeframe.description
            ),
        },
    )


@login_required
def show_etfs(request):
    validate_user(request.user)
    matching_codes = all_etfs()
    extra_context = {
        "title": "Exchange Traded funds over past 300 days",
        "sentiment_heatmap_title": "Sentiment for ETFs",
    }
    return show_companies(
        matching_codes,
        request,
        Timeframe(),
        extra_context,
    )


@login_required
def show_increasing_eps_stocks(request):
    validate_user(request.user)
    matching_companies = increasing_eps(None)
    extra_context = {
        "title": "Stocks with increasing EPS over past 300 days",
        "sentiment_heatmap_title": "Sentiment for selected stocks",
    }
    return show_companies(
        matching_companies,
        request,
        Timeframe(),
        extra_context,
    )


@login_required
def show_increasing_yield_stocks(request):
    validate_user(request.user)
    matching_companies = increasing_yield(None)
    extra_context = {
        "title": "Stocks with increasing yield over past 300 days",
        "sentiment_heatmap_title": "Sentiment for selected stocks",
    }
    return show_companies(
        matching_companies,
        request,
        Timeframe(),
        extra_context,
    )
