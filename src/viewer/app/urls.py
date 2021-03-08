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
from django.urls import path, include
from app.views import *
from django.conf import settings

urlpatterns = [
    path('accounts/', include('django.contrib.auth.urls')),
    path('admin/', admin.site.urls),
    path('', show_all_stocks),
    path('search/by-sector', sector_search),
    path('search/by-yield', dividend_search),
    path('search/by-company', company_search),
    path('search/movers', mover_search),
    path('show/increasing-eps', show_increasing_eps_stocks),  # NB: order important here!
    path('show/increasing-yield', show_increasing_yield_stocks),
    path('show/trends', show_trends),
    path('show/purchase-performance', show_purchase_performance),
    path('show/watched', show_watched, name='show-watched'),
    path('show/etfs', show_etfs, name='show-etfs'),
    path('show/<str:stock>', show_stock, name='show-stock'),
    path('show/outliers/sector/<int:sector_id>/<int:n_days>', show_sector_outliers, name='show-sector-outliers'),
    path('show/outliers/watchlist/<int:n_days>', show_watchlist_outliers, name='show-watchlist-outliers'), # eg. show/outliers/{all,watchlist} WARNING: slow
    path('watchlist/<str:stock>', toggle_watched),
    path('purchase/<str:stock>', buy_virtual_stock),
    path('update/purchase/<slug:slug>', edit_virtual_stock),
    path('delete/purchase/<slug:slug>', delete_virtual_stock),
    path('stats/market-sentiment', market_sentiment),
    path('data/<slug:dataset>/<str:format>/', download_data, name='data'),
    path('show/optimized/watchlist/', show_optimised_watchlist, name='show-optimised-watchlist'),
    path('show/optimized/watchlist/<str:exclude>', show_optimised_watchlist, name='show-optimised-watchlist'),
    path('show/optimized/sector/<int:sector_id>/', show_optimised_sector, name='show-optimised-sector'),
    path('show/optimized/sector/<int:sector_id>/<str:exclude>', show_optimised_sector, name='show-optimised-sector'),
    path('show/optimized/etfs/', show_optimised_etfs, name='show-optimised-etfs'),
    path('show/optimized/etfs/<str:exclude>', show_optimised_etfs, name='show-optimised-etfs')
]


if settings.DEBUG:
    import debug_toolbar
    urlpatterns = [
        path('__debug__/', include(debug_toolbar.urls)),
    ] + urlpatterns
