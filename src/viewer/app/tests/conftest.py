from pytest_factoryboy import register

from app.factories import CompanyDetailsFactory, QuotationFactory

register(CompanyDetailsFactory)
register(QuotationFactory)
