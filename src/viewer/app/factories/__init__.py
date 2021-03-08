import factory
from pytest_factoryboy import register

class CompanyDetailsFactory(factory.django.DjangoModelFactory):
     class Meta:
         model = 'app.CompanyDetails'

     sector_name = factory.Sequence(lambda n: 'Financials')
     asx_code = factory.Sequence(lambda n: 'ANZ')
