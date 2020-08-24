from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from django.views.generic import FormView
from django.http import HttpResponseRedirect, Http404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.views.generic.list import MultipleObjectTemplateResponseMixin, MultipleObjectMixin
from app.models import *
from app.mixins import SearchMixin
from app.forms import SectorSearchForm, DividendSearchForm, CompanySearchForm
from app.analysis import relative_strength, analyse_sector, heatmap_market
from app.plots import *

class SectorSearchView(SearchMixin, LoginRequiredMixin, MultipleObjectMixin, MultipleObjectTemplateResponseMixin, FormView):
    form_class = SectorSearchForm
    template_name = "search_form.html" # generic template, not specific to this view
    action_url = '/search/by-sector'
    paginate_by = 50
    ordering = ('-annual_dividend_yield', 'asx_code') # keep pagination happy, but not used by get_queryset()
    sector_name = None
    as_at_date = None

    def render_to_response(self, context):
        n_days = 7
        if len(self.sector_name) > 0:
            sentiment_bin_df, top10, bottom10, n_stocks = heatmap_sector(self.sector_name, n_days)
            fig = make_sentiment_plot(sentiment_bin_df)
            sentiment_data = plot_as_base64(fig).decode('utf-8')
            plt.close(fig)
        else:
            sector_sentiment = None
        context.update({ # NB: we use update() to not destroy page_obj
            'sector': self.sector_name,
            'most_recent_date': self.as_at_date,
            'watched': user_watchlist(self.request.user),
            'sentiment_heatmap': sentiment_data,
            'sentiment_heatmap_title': "Recent sentiment for {}: past {} days".format(self.sector_name, n_days)
        })
        return super().render_to_response(context)

    def get_queryset(self, **kwargs):
       if kwargs == {}:
           return Quotation.objects.none()
       assert 'sector' in kwargs and len(kwargs['sector']) > 0
       self.sector_name = kwargs['sector']
       all_dates = all_available_dates()
       wanted_stocks = set(sector_stocks(self.sector_name))

       if any(['best10' in kwargs, 'worst10' in kwargs]):
           sector_df, b10, w10 = analyse_sector(self.sector_name)
           wanted = set()
           restricted = False
           if kwargs.get('best10', False):
               wanted = wanted.union(b10.index)
               restricted = True
           if kwargs.get('worst10', False):
               wanted = wanted.union(w10.index)
               restricted = True
           if restricted:
               wanted_stocks = wanted_stocks.intersection(wanted)
           # FALLTHRU...
       self.as_at_date = all_dates[-1]
       print("Looking for {} companies as at {}".format(len(wanted_stocks), self.as_at_date))
       results = company_quotes(wanted_stocks, required_date=self.as_at_date)
       results = results.order_by(*self.ordering)
       return results

sector_search = SectorSearchView.as_view()

class DividendYieldSearch(SearchMixin, LoginRequiredMixin, MultipleObjectMixin, MultipleObjectTemplateResponseMixin, FormView):
    form_class = DividendSearchForm
    template_name = "search_form.html" # generic template, not specific to this view
    action_url = '/search/by-yield'
    paginate_by = 50
    ordering = ('-annual_dividend_yield', 'asx_code') # keep pagination happy, but not used by get_queryset()
    as_at_date = None

    def render_to_response(self, context):
        context.update({
           'most_recent_date': self.as_at_date,
           'watched': user_watchlist(self.request.user)
        })
        return super().render_to_response(context)

    def get_queryset(self, **kwargs):
        if kwargs == {}:
            return Quotation.objects.none()

        self.as_at_date = latest_quotation_date('ANZ')
        min_yield = kwargs.get('min_yield') if 'min_yield' in kwargs else 0.0
        max_yield = kwargs.get('max_yield') if 'max_yield' in kwargs else 10000.0
        results = Quotation.objects.filter(fetch_date=self.as_at_date) \
                              .filter(annual_dividend_yield__gte=min_yield) \
                              .filter(annual_dividend_yield__lte=max_yield)
        if 'min_pe' in kwargs:
            results = results.filter(pe__gte=kwargs.get('min_pe'))
        if 'max_pe' in kwargs:
            results = results.filter(pe__lt=kwargs.get('max_pe'))

        results = results.order_by(*self.ordering)
        return results

dividend_search = DividendYieldSearch.as_view()


class CompanySearch(DividendYieldSearch):
    form_class = CompanySearchForm
    action_url = "/search/by-company"
    paginate_by = 50

    def get_queryset(self, **kwargs):
        if kwargs == {} or not any(['name' in kwargs, 'activity' in kwargs]):
            return Quotation.objects.none()

        self.as_at_date = latest_quotation_date('ANZ')
        matching_companies = set()
        wanted_name = kwargs.get('name', '')
        wanted_activity = kwargs.get('activity', '')
        if len(wanted_name) > 0:
            # match by company name first...
            matching_companies.update([hit.asx_code for hit in
                                       CompanyDetails.objects.filter(name_full__icontains=wanted_name)])
            # but also matching codes
            matching_companies.update([hit.asx_code for hit in
                                       CompanyDetails.objects.filter(asx_code__icontains=wanted_name)])
        if len(wanted_activity) > 0:
            matching_companies.update([hit.asx_code for hit in \
                                       CompanyDetails.objects.filter(principal_activities__icontains=wanted_activity)])
        print("Showing results for {} companies".format(len(matching_companies)))
        results = Quotation.objects.filter(fetch_date=self.as_at_date) \
                                   .filter(asx_code__in=matching_companies) \
                                   .order_by(*self.ordering)
        return results

company_search = CompanySearch.as_view()

@login_required
def all_stocks(request):
   all_dates = all_available_dates()
   if len(all_dates) < 1:
       raise Http404("No ASX price data available!")
   ymd = all_dates[-1]
   assert isinstance(ymd, str) and len(ymd) > 8
   qs = Quotation.objects.filter(fetch_date=ymd) \
                         .exclude(asx_code__isnull=True) \
                         .exclude(last_price__isnull=True) \
                         .exclude(volume=0) \
                         .order_by('-annual_dividend_yield', '-last_price', '-volume')
   assert qs is not None
   paginator = Paginator(qs, 50)
   page_number = request.GET.get('page', 1)
   page_obj = paginator.get_page(page_number)
   context = {
       "page_obj": page_obj,
       "most_recent_date": ymd,
       "watched": user_watchlist(request.user)
   }
   return render(request, "all_stocks.html", context=context)

def show_stock(request, stock=None):
   validate_stock(stock)

   # make stock momentum plot
   quotes = Quotation.objects.filter(asx_code=stock).exclude(last_price__isnull=True)
   securities = Security.objects.filter(asx_code=stock)
   company_details = CompanyDetails.objects.filter(asx_code=stock).first()
   if company_details is None:
       raise Http404("No company details for {}".format(stock))
   df = as_dataframe(quotes)
   #print(df['last_price'])
   if len(df) < 14:  # RSI requires at least 14 prices to plot so reject recently added stocks
       raise Http404("Insufficient price quotes for {}".format(stock))
   fig = make_rsi_plot(stock, df)
   rsi_data = plot_as_base64(fig)
   plt.close(fig)

   # show sector performance over past 3 months
   tag = "sector_momentum-{}".format(company_details.sector_name)
   cache_hit = ImageCache.objects.filter(tag=tag).first()
   if cache_hit is None or cache_hit.is_outdated():
       sector_df, b10, w10 = analyse_sector(company_details.sector_name)
       #print(sector_df)
       fig = make_sector_momentum_plot(sector_df, company_details.sector_name)
       sector_b64 = plot_as_base64(fig).decode('utf-8')
       plt.close(fig)
       update_image_cache(tag, sector_b64)
   else:
       sector_b64 = cache_hit.base64

   # populate template and render HTML page with context
   context = {
       'rsi_data': rsi_data.decode('utf-8'),
       'asx_code': stock,
       'securities': securities,
       'cd': company_details,
       'sector_past3months_data': sector_b64,
   }
   return render(request, "stock_view.html", context=context)

@login_required
def market_sentiment(request):
    n_days = 7
    sentiment_bin_df, top10, bottom10, n_stocks = heatmap_market(n_days) # need to capture at least last 5 trading days, no matter what day it is run on
    fig = make_sentiment_plot(sentiment_bin_df)
    sentiment_data = plot_as_base64(fig)
    plt.close(fig)

    context = {
       'sentiment_data': sentiment_data.decode('utf-8'),
       'n_days': n_days,
       'n_stocks_plotted': n_stocks,
       'best_ten': top10, # NB: each day
       'worst_ten': bottom10,
       'watched': user_watchlist(request.user)
    }
    return render(request, 'market_sentiment_view.html', context=context)

@login_required
def show_watched(request):
    matching_companies = user_watchlist(request.user)
    print("Showing results for {} companies".format(len(matching_companies)))
    stocks = company_quotes(matching_companies)
    purchases = user_purchases(request.user)

    # paginate results for 50 stocks per page
    paginator = Paginator(stocks, 50)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.page(page_number)

    # add sentiment heatmap amongst watched stocks
    n_days = 14
    sector_heatmap_data = None
    if len(matching_companies) > 0:
        sentiment_bin_df, top10, bottom10, n_stocks = heatmap_companies(matching_companies, n_days=n_days)
        fig = make_sentiment_plot(sentiment_bin_df)
        sentiment_heatmap_data = plot_as_base64(fig).decode('utf-8')
        plt.close(fig)

    context = {
         "most_recent_date": latest_quotation_date('ANZ'),
         "page_obj": page_obj,
         "title": "Stocks you are watching",
         "watched": user_watchlist(request.user),
         "virtual_purchases": purchases,
         "sentiment_heatmap": sentiment_heatmap_data,
         "sentiment_heatmap_title": "Watched stock sentiment heatmap: past {} days".format(n_days)
    }
    return render(request, 'all_stocks.html', context=context)

def redirect_to_next(request, fallback_next='/'):
    # redirect will trigger a redraw which will show the purchase since next will be the same page
    assert request is not None
    if request.GET is not None:
        return HttpResponseRedirect(request.GET.get('next', fallback_next))
    else:
        return HttpResponseRedirect(fallback_next)

@login_required
def toggle_watched(request, stock=None):
    validate_stock(stock)
    current_watchlist = user_watchlist(request.user)
    if stock in current_watchlist: # remove from watchlist?
        Watchlist.objects.filter(user=request.user, asx_code=stock).delete()
    else:
        w = Watchlist(user=request.user, asx_code=stock)
        w.save()
    return redirect_to_next(request)

@login_required
def buy_virtual_stock(request, stock=None, amount=5000.0):
    validate_stock(stock)
    assert amount > 0.0
    quote, latest_date = latest_quote(stock)
    cur_price = quote.last_price
    if cur_price >= 1e-6:
        vp = VirtualPurchase(asx_code=stock, user=request.user,
                             buy_date=latest_date, price_at_buy_date=cur_price,
                             amount=amount, n=int(amount / cur_price))
        vp.save()
        print("Purchased {} as at {}: {} shares at {} each".format(vp.asx_code, latest_date, vp.n, vp.price_at_buy_date))
    else:
        print("WARNING: cant buy {} as its price is zero".format(stock))
        # do nothing in this case...
        pass
    return redirect_to_next(request)

@login_required
def delete_virtual_stock(request, buy_date=None, stock=None):
    validate_stock(stock)
    validate_date(buy_date)

    stocks_to_delete = VirtualPurchase.objects.filter(buy_date=buy_date, asx_code=stock)
    print("Selected {} stocks to delete".format(len(stocks_to_delete)))
    stocks_to_delete.delete()
    return redirect_to_next(request)
