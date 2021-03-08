import factory
from pytest_factoryboy import register
import faker

faker = faker.Faker()

class CompanyDetailsFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'app.CompanyDetails'

    sector_name = factory.Sequence(lambda n: 'Financials')
    asx_code = factory.Sequence(lambda n: 'ANZ')

class QuotationFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'app.Quotation'
    
    asx_code = factory.Sequence(lambda n: 'ANZ')
    error_code = '' # no error
    fetch_date = faker.date()