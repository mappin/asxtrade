import factory

from app.models import CompanyDetails


class CompanyDetailsFactory(factory.django.DjangoModelFactory):
  class Meta:
       model = CompanyDetails

  sector_name = factory.Sequence(lambda n: 'Financials')
  asx_code = factory.Sequence(lambda n: 'ANZ')
