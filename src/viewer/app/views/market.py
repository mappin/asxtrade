"""
Responsible for handling requests for pages from the website and delegating the analysis
and visualisation as required.
"""
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from app.models import (
    user_watchlist,
    latest_quotation_date,
    cached_all_stocks_cip,
    Timeframe,
    validate_user
)
from app.plots import (
    plot_heatmap,
    plot_market_wide_sector_performance,
    plot_market_cap_distribution,
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

