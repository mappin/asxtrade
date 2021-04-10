from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from app.models import (user_watchlist, validate_user, Timeframe, validate_stock, toggle_watchlist_entry)
from app.views.core import show_companies

@login_required
def show_watched(request):
    validate_user(request.user)
    matching_companies = user_watchlist(request.user)

    timeframe = Timeframe()
    return show_companies(
        matching_companies,
        request,
        timeframe,
        {
            "title": "Stocks you are watching",
            "sentiment_heatmap_title": "Watchlist stocks sentiment: {}".format(timeframe.description),
        }
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
    toggle_watchlist_entry(request.user, stock)
    return redirect_to_next(request)
