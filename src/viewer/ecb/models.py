"""Data models for the European Central Bank datasets"""
from django.db import models
from djongo.models import ObjectIdField, DjongoManager


class ECBMetadata(models.Model):
    """
    Model responsible for describing allowable values for a given metadata_type (dimension, attribute, measure)
    which is used for form generation and validation
    """

    id = ObjectIdField(db_column="_id", primary_key=True)
    metadata_type = models.TextField(db_column="metadata_type", null=False, blank=False)
    flow = models.TextField(db_column="flow_name", null=False, blank=False)
    code = models.TextField(db_column="item_name", null=False, blank=False)
    column_name = models.TextField(db_column="codelist_name", null=False, blank=False)
    printable_code = models.TextField(db_column="item_value", null=False, blank=False)

    objects = DjongoManager()

    class Meta:
        db_table = "ecb_metadata_index"
        verbose_name = "ECB Metadata"
        verbose_name_plural = "ECB Metadata"


class ECBFlow(models.Model):
    """
    Describe each dataflow ingested from ECB SDMX REST API
    """

    id = ObjectIdField(db_column="_id", primary_key=True)
    name = models.TextField(db_column="flow_name", null=False, blank=False)
    description = models.TextField(db_column="flow_descr", null=False, blank=False)
    is_test_data = models.BooleanField()
    last_updated = models.DateTimeField()
    prepared = models.DateTimeField()
    sender = models.TextField()
    source = models.TextField()

    objects = DjongoManager()

    class Meta:
        db_table = "ecb_flow_index"
        verbose_name = "ECB Flow"
        verbose_name_plural = "ECB Flow"


class ECBDataCache(models.Model):
    """ Similar to worldbank data cache """

    id = ObjectIdField(db_column="_id", primary_key=True)
    size_in_bytes = models.IntegerField()
    status = models.TextField()
    tag = models.TextField()
    dataframe_format = models.TextField()
    field = models.TextField()
    last_updated = models.DateTimeField()
    market = models.TextField()
    n_days = models.IntegerField()
    n_stocks = models.IntegerField()
    sha256 = models.TextField()
    scope = models.TextField()
    dataframe = models.BinaryField()

    objects = DjongoManager()

    class Meta:
        db_table = "ecb_data_cache"
        verbose_name = "ECB Data Cache"
        verbose_name_plural = "ECB Data Cache"