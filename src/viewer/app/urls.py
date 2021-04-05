"""viewer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf import settings
from django.urls import path, include
from app.views import (
    show_all_stocks,
    sector_search,
    dividend_search,
    company_search,
    mover_search,
    market_cap_search,
    show_increasing_eps_stocks,
    show_increasing_yield_stocks,
    show_trends,
    show_purchase_performance,
    show_recent_sector,
    show_watched,
    show_etfs,
    show_stock,
    show_fundamentals,
    show_stock_sector,
    show_sector_outliers,
    show_watchlist_outliers,
    toggle_watched,
    buy_virtual_stock,
    edit_virtual_stock,
    delete_virtual_stock,
    market_sentiment,
    download_data,
    optimised_watchlist_view,
    optimised_etf_view,
    optimised_sector_view
)

urlpatterns = [
    path("accounts/", include("django.contrib.auth.urls")),
    path("admin/", admin.site.urls),
    path("", show_all_stocks),
    path("search/by-sector", sector_search),
    path("search/by-yield", dividend_search),
    path("search/by-company", company_search),
    path("search/movers", mover_search),
    path("search/market-cap", market_cap_search),
    path(
        "show/increasing-eps", show_increasing_eps_stocks
    ),  # NB: order important here!
    path("show/increasing-yield", show_increasing_yield_stocks),
    path("show/trends", show_trends),
    path("show/purchase-performance", show_purchase_performance),
    path("show/watched", show_watched, name="show-watched"),
    path("show/etfs", show_etfs, name="show-etfs"),
    path("show/fundamentals/<str:stock>", show_fundamentals),
    path("show/recent_sector_performance", show_recent_sector, name="recent-sector-performance"),
    path("show/stock_sector/<str:stock>", show_stock_sector),
    path("show/<str:stock>", show_stock, name="show-stock"),
    path(
        "show/outliers/sector/<int:sector_id>/<int:n_days>",
        show_sector_outliers,
        name="show-sector-outliers",
    ),
    path(
        "show/outliers/watchlist/<int:n_days>",
        show_watchlist_outliers,
        name="show-watchlist-outliers",
    ),  # eg. show/outliers/{all,watchlist} WARNING: slow
    path("watchlist/<str:stock>", toggle_watched),
    path("purchase/<str:stock>", buy_virtual_stock),
    path("update/purchase/<slug:slug>", edit_virtual_stock),
    path("delete/purchase/<slug:slug>", delete_virtual_stock),
    path("stats/market-sentiment", market_sentiment),
    path("data/<slug:dataset>/<str:output_format>/", download_data, name="data"),
    path(
        "show/optimized/watchlist/",
        optimised_watchlist_view,
        name="show-optimised-watchlist",
    ),
    path(
        "show/optimized/sector/",
        optimised_sector_view,
        name="show-optimised-sector",
    ),
    path("show/optimized/etfs/", optimised_etf_view, name="show-optimised-etfs"),

]


if settings.DEBUG:
    import debug_toolbar
    urlpatterns = [path("__debug__/", include(debug_toolbar.urls)),] + urlpatterns
