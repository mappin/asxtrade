"""
Responsible for the core view - showing lists (paginated) of stocks handling all required corner conditions for missing data.
Other views are derived from this (both FBV and CBV) so be careful to take sub-views needs into account here.
"""
from django.db.models.query import QuerySet
from django.core.paginator import Paginator
from django.shortcuts import render
from django.http import Http404
from django.contrib.auth.decorators import login_required
from app.models import (
    Timeframe, user_purchases, latest_quote, user_watchlist,
    selected_cached_stocks_cip, valid_quotes_only,
    all_available_dates, validate_date, validate_user, 
    all_etfs, increasing_yield, increasing_eps
)
from app.messages import info, warning, add_messages
from app.plots import plot_heatmap, plot_breakdown

def show_companies(
        matching_companies, # may be QuerySet or iterable of stock codes (str)
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
    sort_by = tuple(request.GET.get("sort_by", "asx_code").split(","))
    info(request, "Sorting by {}".format(sort_by))
    stocks_queryset = stocks_queryset.order_by(*sort_by)

    # keep track of stock codes for template convenience
    asx_codes = [quote.asx_code for quote in stocks_queryset.all()]
    n_top_bottom = extra_context['n_top_bottom'] if 'n_top_bottom' in extra_context else 20
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
    assert isinstance(stocks_queryset, QuerySet)
    paginator = Paginator(stocks_queryset, 50)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.page(page_number)
    context['page_obj'] = page_obj
    context['object_list'] = paginator

    if len(asx_codes) <= 0:
        warning(request, "No matching companies found.")
    else:
        df = selected_cached_stocks_cip(asx_codes, sentiment_timeframe)
        sentiment_heatmap_data, top10, bottom10 = plot_heatmap(df, sentiment_timeframe, n_top_bottom=n_top_bottom)
        sector_breakdown_plot = plot_breakdown(df)
        context.update({
            "best_ten": top10,
            "worst_ten": bottom10,
            "sentiment_heatmap": sentiment_heatmap_data,
            "sentiment_heatmap_title": "{}: {}".format(context['title'], sentiment_timeframe.description),
            "sector_breakdown_plot": sector_breakdown_plot,
        })

    if extra_context:
        context.update(extra_context)
    add_messages(request, context)
    #print(context)
    return render(request, template_name, context=context)


@login_required
def show_all_stocks(request):
    all_dates = all_available_dates()
    if len(all_dates) < 1:
        raise Http404("No ASX price data available!")
    ymd = all_dates[-1]
    validate_date(ymd)
    qs = valid_quotes_only(ymd)
    timeframe = Timeframe()
    return show_companies(qs, request, timeframe, extra_context={
        "title": "All stocks",
        "sentiment_heatmap_title": "All stock sentiment: {}".format(timeframe.description)
    })


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
