from pytest_factoryboy import register

from app.factories import CompanyDetailsFactory, QuotationFactory, SecurityFactory

register(CompanyDetailsFactory)
register(QuotationFactory)
register(SecurityFactory)