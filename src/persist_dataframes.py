#!/usr/bin/python3
import pymongo
from bson.binary import Binary
import argparse
import io
import os
import pandas as pd
import numpy
from datetime import datetime, date
import calendar
import hashlib

def dates_of_month(month, year):
    assert month >= 1 and month <= 12
    assert year >= 2020
    num_days = calendar.monthrange(year, month)[1]
    days = [str(date(year, month, day)) for day in range(1, num_days+1)]
    #print(month, year, " ", days)
    return days

def clean_value(value):
    """
    To ensure the computed dataframe is consistently typed all values are
    co-erced to float or we die trying...
    """
    if isinstance(value, (int, float)):
        return float(value)
    # else str...
    value = value.rstrip('%')
    value = value.replace(',', '')
    return float(value)

def load_prices(db, field_name, month, year):
    """
    Build a matrix of all available stocks with the given field_name eg. 'last_price'. This
    is built into a pandas dataframe and returned. NaN's may be returned where the stock has no quotation at a given date.
    All dates in the specified month are included in the dataframe.
    The dataframe is always organised as stock X dates (stocks are rows) with the values of the specified field as float values
    """
    assert db is not None
    assert len(field_name) > 0
    days_of_month = dates_of_month(month, year)
    rows = [{ 'asx_code': row['asx_code'],
              'fetch_date': row['fetch_date'],
              field_name: clean_value(row[field_name])}
              for row in db.asx_prices.find({ 'fetch_date': { "$in": days_of_month }, field_name: { "$exists": True } }, { 'asx_code': 1, field_name: 1, 'fetch_date': 1})]
    if len(rows) == 0:
        df = pd.DataFrame(columns=['fetch_date', 'asx_code', field_name]) # return dummy dataframe if empty
        return df
        # FALLTHRU
    df = pd.DataFrame.from_records(rows)
    df = df.pivot(index='asx_code', columns='fetch_date', values=field_name)
    return df

def load_all_prices(db, month, year, status='FINAL', market='asx', scope='all-downloaded'):
    for field_name in ['change_in_percent', 'last_price', 'change_price', 'day_low_price', 'day_high_price', 'volume', 'eps', 'pe', 'annual_dividend_yield']:
        print("Constructing matrix: {} {}-{}".format(field_name, month, year))
        df = load_prices(db, field_name, month, year)

        with io.BytesIO() as fp:
             # NB: if this fails it may be because you are using fastparquet which doesnt (yet) support BytesIO. Use eg. pyarrow
             df.to_parquet(fp, compression='gzip', index=True)
             fp.seek(0)
             bytes = fp.read()
             tag = "{}-{:02d}-{}-{}".format(field_name, month, year, market)
             db.market_quote_cache.update_one({ 'tag': tag, 'scope': scope}, { "$set": {
                     'tag': tag, 'status': status,
                     'last_updated': datetime.utcnow(),
                     'field': field_name,
                     'market': market,
                     'scope': scope,
                     'n_days': len(df.columns),
                     'n_stocks': len(df),
                     'dataframe_format': 'parquet',
                     'size_in_bytes': len(bytes),
                     'sha256': hashlib.sha256(bytes).hexdigest(),
                     'dataframe': Binary(bytes), # NB: always parquet format
                 }}, upsert=True)

if __name__ == "__main__":
   a = argparse.ArgumentParser(description="Construct and ingest db.asx_prices into parquet format month-by-month and persist to mongo")
   default_host = 'pi1'
   default_port = 27017
   default_db = 'asxtrade'
   default_access = 'read-write'
   default_user = 'rw'
   a.add_argument("--db",
                     help="Mongo host/ip to save to [{}]".format(default_host),
                     type=str, default=default_host)
   a.add_argument("--port",
                     help="TCP port to access mongo db [{}]".format(str(default_port)),
                     type=int, default=default_port)
   a.add_argument("--dbname",
                     help="Name on mongo DB to access [{}]".format(default_db),
                     type=str, default=default_db)
   dict_args = { 'help': "MongoDB RBAC username to use ({} access required) [{}]".format(default_access, default_user),
                 'type': str,
                 'required': True
               }
   if default_user is not None:
       dict_args.pop('required', None)
       dict_args.update({ 'default': default_user })
   a.add_argument("--dbuser", **dict_args)
   a.add_argument("--dbpassword", help="MongoDB password for user", type=str, required=True)
   a.add_argument("--month", help="Month of year 1..12", required=True, type=int)
   a.add_argument("--year", help="Year to load [2020]", default=2020, type=int)
   a.add_argument("--status", help="Status of matrix eg. INCOMPLETE or FINAL", required=True, type=str)
   args = a.parse_args()

   pwd = str(args.dbpassword)
   if pwd.startswith('$'):
       pwd = os.getenv(args.dbpassword[1:])
   mongo = pymongo.MongoClient(args.db, args.port, username=args.dbuser, password=pwd)
   db = mongo[args.dbname]

   load_all_prices(db, args.month, args.year, args.status)
   print("Run completed successfully.")
   exit(0)
