"""
Responsible for implementing user search for finding companies of interest
"""
from collections import defaultdict
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.list import MultipleObjectTemplateResponseMixin
from django.views.generic.edit import FormView
from django.shortcuts import render
import pandas as pd
from pandas.core.base import NoNewAttributesMixin
from app.mixins import SearchMixin
from app.data import cache_plot, timeframe_end_performance
from app.messages import info
from app.forms import (
    DividendSearchForm,
    SectorSearchForm,
    MoverSearchForm,
    CompanySearchForm,
    MarketCapSearchForm,
    SectorSentimentSearchForm,
    FinancialMetricSearchForm,
)
from app.models import (
    Timeframe,
    Quotation,
    latest_quotation_date,
    valid_quotes_only,
    latest_quote,
    Sector,
    all_sector_stocks,
    find_movers,
    find_named_companies,
    selected_cached_stocks_cip,
    CompanyFinancialMetric,
)
from app.data import make_sector_performance_dataframe
from app.views.core import show_companies
from app.plots import plot_sector_performance, plot_boxplot_series
from app.messages import warning
from lazydict import LazyDictionary
from plotnine.layer import Layers


class DividendYieldSearch(
    SearchMixin,
    LoginRequiredMixin,
    MultipleObjectTemplateResponseMixin,
    FormView,
):
    form_class = DividendSearchForm
    template_name = "search_form.html"  # generic template, not specific to this view
    action_url = "/search/by-yield"
    ordering = (
        "asx_code"  # default order must correspond to default JS code in template
    )
    timeframe = Timeframe(past_n_days=30)
    qs = None

    def additional_context(self, context):
        """
        Return the additional fields to be added to the context by render_to_response(). Subclasses
        should override this rather than the template design pattern implementation of render_to_response()
        """
        assert context is not None
        if self.timeframe is not None:
            info(
                self.request,
                f"Unless stated otherwise, timeframe is {self.timeframe.description}",
            )
        return {
            "title": "Find by dividend yield or P/E",
            "timeframe": self.timeframe,
            "sentiment_heatmap_title": "Matching stock heatmap",
            "n_top_bottom": 20,
        }

    def render_to_response(self, context, **kwargs):
        """
        Invoke show_companies()
        """
        assert kwargs is not None
        context.update(self.additional_context(context))

        return show_companies(  # will typically invoke show_companies() to share code across all views
            self.qs,
            self.request,
            self.timeframe,
            context,
            template_name=self.template_name,
        )

    def get_queryset(self, **kwargs):
        if kwargs == {}:
            return Quotation.objects.none()

        as_at = latest_quotation_date("ANZ")
        min_yield = kwargs.get("min_yield") if "min_yield" in kwargs else 0.0
        max_yield = kwargs.get("max_yield") if "max_yield" in kwargs else 10000.0
        results = (
            Quotation.objects.filter(fetch_date=as_at)
            .filter(annual_dividend_yield__gte=min_yield)
            .filter(annual_dividend_yield__lte=max_yield)
        )

        if "min_pe" in kwargs:
            results = results.filter(pe__gte=kwargs.get("min_pe"))
        if "max_pe" in kwargs:
            results = results.filter(pe__lt=kwargs.get("max_pe"))
        if "min_eps_aud" in kwargs:
            results = results.filter(eps__gte=kwargs.get("min_eps_aud"))
        self.qs = results
        return self.qs


dividend_search = DividendYieldSearch.as_view()


class SectorSearchView(DividendYieldSearch):
    DEFAULT_SECTOR = "Communication Services"
    form_class = SectorSearchForm
    action_url = "/search/by-sector"
    template_name = "sector_search_form.html"
    ld = LazyDictionary()

    def additional_context(self, context):
        d = super().additional_context(context)
        ld = self.ld
        sector = ld.get(
            "sector", self.DEFAULT_SECTOR
        )  # default to Comms. Services if not specified'
        sector_id = ld.get("sector_id", None)
        d.update(
            {
                # to highlight top10/bottom10 bookmarks correctly
                "title": "Find by company sector",
                "sector_name": sector,
                "sector_id": sector_id,
                "sentiment_heatmap_title": "{} sector sentiment".format(sector),
                "sector_performance_plot_uri": ld.get("sector_performance_plot", None),
                "timeframe_end_performance": timeframe_end_performance(ld),
            }
        )
        return d

    def get_queryset(self, **kwargs):
        def sector_performance(ld: LazyDictionary) -> str:
            sector = ld.get("sector")
            return cache_plot(
                f"{sector}-sector-performance",
                lambda ld: plot_sector_performance(ld["sector_performance_df"], sector)
                if ld["sector_performance_df"] is not None
                else None,
                datasets=ld,
                dont_cache=True,
            )

        # user never run this view before?
        if kwargs == {}:
            print("WARNING: no form parameters specified - returning empty queryset")
            return Quotation.objects.none()

        sector = kwargs.get("sector", self.DEFAULT_SECTOR)
        sector_id = int(Sector.objects.get(sector_name=sector).sector_id)
        wanted_stocks = all_sector_stocks(sector)
        print("Found {} stocks matching sector={}".format(len(wanted_stocks), sector))
        report_top_n = kwargs.get("report_top_n", None)
        report_bottom_n = kwargs.get("report_bottom_n", None)
        self.timeframe = Timeframe(past_n_days=90)
        ld = LazyDictionary()
        ld["sector"] = sector
        ld["sector_id"] = sector_id
        ld["sector_companies"] = wanted_stocks
        ld["cip_df"] = selected_cached_stocks_cip(wanted_stocks, self.timeframe)
        ld["sector_performance_df"] = lambda ld: make_sector_performance_dataframe(
            ld["cip_df"], ld["sector_companies"]
        )
        ld["sector_performance_plot"] = lambda ld: sector_performance(ld)
        self.ld = ld
        if report_top_n is not None or report_bottom_n is not None:
            cip_sum = self.ld["cip_df"].transpose().sum().to_frame(name="percent_cip")

            # print(cip_sum)
            top_N = (
                set(cip_sum.nlargest(report_top_n, "percent_cip").index)
                if report_top_n is not None
                else set()
            )
            bottom_N = (
                set(cip_sum.nsmallest(report_bottom_n, "percent_cip").index)
                if report_bottom_n is not None
                else set()
            )
            wanted_stocks = top_N.union(bottom_N)
        print("Requesting valid quotes for {} stocks".format(len(wanted_stocks)))
        quotations_as_at, actual_mrd = valid_quotes_only(
            "latest", ensure_date_has_data=True
        )
        self.qs = quotations_as_at.filter(asx_code__in=wanted_stocks)
        if len(self.qs) < len(wanted_stocks):
            got = set([q.asx_code for q in self.qs.all()])
            missing_stocks = wanted_stocks.difference(got)
            warning(
                self.request,
                f"could not obtain quotes for all stocks as at {actual_mrd}: {missing_stocks}",
            )
        return self.qs


sector_search = SectorSearchView.as_view()


class MoverSearch(DividendYieldSearch):
    form_class = MoverSearchForm
    action_url = "/search/movers"

    def additional_context(self, context):
        d = super().additional_context(context)
        d.update(
            {
                "title": "Find companies exceeding threshold over timeframe",
                "sentiment_heatmap_title": "Heatmap for moving stocks",
            }
        )
        return d

    def get_queryset(self, **kwargs):
        if any(
            [kwargs == {}, "threshold" not in kwargs, "timeframe_in_days" not in kwargs]
        ):
            return Quotation.objects.none()
        threshold_percentage = kwargs.get("threshold")
        self.timeframe = Timeframe(past_n_days=kwargs.get("timeframe_in_days", 30))

        df = find_movers(
            threshold_percentage,
            self.timeframe,
            kwargs.get("show_increasing", False),
            kwargs.get("show_decreasing", False),
            kwargs.get("max_price", None),
            field=kwargs.get("metric", "change_in_percent"),
        )
        # print(df)

        self.qs, _ = latest_quote(tuple(df.index))
        return self.qs


mover_search = MoverSearch.as_view()


class CompanySearch(DividendYieldSearch):
    form_class = CompanySearchForm
    action_url = "/search/by-company"

    def additional_context(self, context):
        return {
            "title": "Find by company name or activity",
            "sentiment_heatmap_title": "Heatmap for named companies",
        }

    def get_queryset(self, **kwargs):
        if kwargs == {} or not any(["name" in kwargs, "activity" in kwargs]):
            return Quotation.objects.none()
        wanted_name = kwargs.get("name", "")
        wanted_activity = kwargs.get("activity", "")
        matching_companies = find_named_companies(wanted_name, wanted_activity)
        print("Showing results for {} companies".format(len(matching_companies)))
        self.qs, _ = latest_quote(tuple(matching_companies))
        return self.qs


company_search = CompanySearch.as_view()


class FinancialMetricSearchView(LoginRequiredMixin, FormView):
    form_class = FinancialMetricSearchForm
    template_name = "search_form.html"
    action_url = "/search/by-metric"

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()
        return form_class(
            [
                (v, v)
                for v in sorted(
                    list(
                        CompanyFinancialMetric.objects.order_by("name")
                        .values_list("name", flat=True)
                        .distinct()
                    )
                )
            ],
            **self.get_form_kwargs(),
        )

    def form_valid(self, form):
        """
        Invoke show_companies()
        """
        wanted_metric = form.cleaned_data.get("metric", None)
        wanted_value = form.cleaned_data.get("amount", None)
        wanted_unit = form.cleaned_data.get("unit", "")
        mult = 1 if not wanted_unit.endswith("(M)") else 1000 * 1000
        wanted_relation = form.cleaned_data.get("relation", ">=")
        print(f"{wanted_metric} {wanted_value} {mult} {wanted_relation}")

        if all([wanted_metric, wanted_value, wanted_relation]):
            matching_records = CompanyFinancialMetric.objects.filter(name=wanted_metric)
            if wanted_relation == "<=":
                matching_records = matching_records.filter(
                    value__lte=wanted_value * mult
                )
            else:
                matching_records = matching_records.filter(
                    value__gte=wanted_value * mult
                )
            matching_stocks = set([m.asx_code for m in matching_records])
            # its possible that some stocks have been delisted at the latest date, despite the financials being a match.
            # They will be shown if the ASX data endpoint still returns data as at the latest quote date
        else:
            matching_stocks = Quotation.objects.none()

        context = self.get_context_data()
        context.update({"title": "Find companies by financial metric"})
        return show_companies(
            matching_stocks,
            self.request,
            Timeframe(past_n_days=30),
            context,
            self.template_name,
        )


financial_metric_search = FinancialMetricSearchView.as_view()


class MarketCapSearch(MoverSearch):
    action_url = "/search/market-cap"
    form_class = MarketCapSearchForm

    def additional_context(self, context):
        return {
            "title": "Find companies by market capitalisation",
            "sentiment_heatmap_title": "Heatmap for matching market cap stocks",
        }

    def get_queryset(self, **kwargs):
        # identify all stocks which have a market cap which satisfies the required constraints
        quotes_qs, most_recent_date = latest_quote(None)
        min_cap = kwargs.get("min_cap", 1)
        max_cap = kwargs.get("max_cap", 1000)
        quotes_qs = (
            quotes_qs.exclude(market_cap__lt=min_cap * 1000 * 1000)
            .exclude(market_cap__isnull=True)
            .exclude(market_cap__gt=max_cap * 1000 * 1000)
            .exclude(suspended__isnull=True)
        )
        info(
            self.request,
            "Found {} quotes, as at {}, satisfying market cap criteria".format(
                quotes_qs.count(), most_recent_date
            ),
        )
        self.qs = quotes_qs
        return self.qs


market_cap_search = MarketCapSearch.as_view()


class ShowRecentSectorView(LoginRequiredMixin, FormView):
    template_name = "recent_sector_performance.html"
    form_class = SectorSentimentSearchForm
    action_url = "/show/recent_sector_performance"

    def form_valid(self, form):
        sector = form.cleaned_data.get("sector", "Communication Services")
        norm_method = form.cleaned_data.get("normalisation_method", None)
        n_days = form.cleaned_data.get("n_days", 30)
        ld = LazyDictionary()
        ld["stocks"] = lambda ld: all_sector_stocks(sector)
        ld["timeframe"] = Timeframe(past_n_days=n_days)
        ld["cip_df"] = lambda ld: selected_cached_stocks_cip(
            ld["stocks"], ld["timeframe"]
        )

        context = self.get_context_data()

        def winner_results(df: pd.DataFrame) -> list:
            # compute star performers: those who are above the mean on a given day counted over all days
            count = defaultdict(int)
            avg = df.mean(axis=0)
            for col in df.columns:
                winners = df[df[col] > avg[col]][col]
                for winner in winners.index:
                    count[winner] += 1
            results = []
            for asx_code, n_wins in count.items():
                x = df.loc[asx_code].sum()
                # avoid "dead cat bounce" stocks which fall spectacularly and then post major increases in percentage terms
                if x > 0.0:
                    results.append((asx_code, n_wins, x))
            return list(reversed(sorted(results, key=lambda t: t[2])))

        context.update(
            {
                "title": "Past {} day sector performance: box plot trends".format(
                    n_days
                ),
                "n_days": n_days,
                "sector": sector,
                "plot_uri": cache_plot(
                    f"{sector}-recent-sector-view-{ld['timeframe'].description}-{norm_method}",
                    lambda ld: plot_boxplot_series(
                        ld["cip_df"], normalisation_method=norm_method
                    ),
                    datasets=ld,
                ),
                "winning_stocks": winner_results(ld["cip_df"]),
            }
        )
        return render(self.request, self.template_name, context)


show_recent_sector = ShowRecentSectorView.as_view()
