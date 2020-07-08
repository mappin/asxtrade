#!/usr/bin/python3.8
import pymongo
import argparse
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime, timedelta, timezone
import json
import csv
import olefile
import tempfile
import time
import pandas as pd
import os

retry_strategy = Retry(
    total=10,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"],
    backoff_factor=5
)
retry_adapter = HTTPAdapter(max_retries=retry_strategy)
 
def update_companies(db, config, ensure_indexes=True):
    resp = requests.get(config.get('asx_companies'))
    if ensure_indexes:
        db.companies.create_index([( 'asx_code', pymongo.ASCENDING ) ], unique=True)

    fname = "{}/companies.{}.csv".format(config.get('data_root'), datetime.now().strftime("%Y-%m-%d"))
    with open(fname, 'w+') as csvfp:
        writer = csv.DictWriter(csvfp, fieldnames=['asx_code', 'name', 'sector', 'last_updated'])
        writer.writeheader()
        n = 0
        for line in resp.text.splitlines():
            if any([line.startswith("ASX listed companies"), len(line.strip()) < 1, line.startswith("Company name")]):
                continue
            fields = line.split('","')  # must use 3 chars here as just splitting on comma will break commas used in sector names (for example)
            assert len(fields) == 3
            d = { "asx_code": fields[1].strip('"'),
                  "name": fields[0].strip('"'),
                  "sector": fields[2].strip('"'), 
                  "last_updated": datetime.utcnow() }
            assert len(d.get('asx_code')) >= 3
            assert len(d.get('name')) > 0
            writer.writerow(d)
            db.companies.find_one_and_update({ 'asx_code': d['asx_code'] }, { '$set': d }, upsert=True)
            n += 1
    print("Saved {} companies to {} for validation by great_expectations.".format(n, fname))  

def update_isin(db, config, ensure_indexes=True):
    resp = requests.get(config.get('asx_isin'))
    if ensure_indexes:
         db.asx_isin.create_index([( 'asx_code', pymongo.ASCENDING), ('asx_isin_code', pymongo.ASCENDING) ], unique=True)

    with tempfile.NamedTemporaryFile(delete=False) as content:
         content.write(resp.content)

    fname = "{}/isin.{}.csv".format(config.get('data_root'), datetime.now().strftime('%Y-%m-%d'))
    with open(fname, 'w+') as csvfp:
         writer = csv.DictWriter(csvfp, fieldnames=['asx_code', 'company_name', 'security_name', 'asx_isin_code', 'last_updated'])
         writer.writeheader()
         df = pd.read_excel(content.name)
         n = 0
         for row in df[4:].itertuples(): # TODO FIXME...first four rows are rubbish - but maybe need to be a bit more flexible
             row_index, asx_code, company_name, security_name, isin_code = row
             asx_code = str(asx_code) # some ASX codes are all integers which we dont want treated as int
             assert len(asx_code) >= 3
             assert len(company_name) > 0
             assert len(security_name) > 0
             d = { 'asx_code': asx_code, 'company_name': company_name, 'security_name': security_name, 
                   'asx_isin_code': isin_code, 'last_updated': datetime.utcnow() }
             writer.writerow(d)
             db.asx_isin.find_one_and_update({ 'asx_isin_code': isin_code, 'asx_code': asx_code }, { "$set": d }, upsert=True)
             n += 1
         print("Saved {} securities to {} for validation by great_expectations.".format(n, fname))

def get_fetcher():
    fetcher = requests.Session()
    fetcher.mount("https://", retry_adapter)
    fetcher.mount("http://", retry_adapter)
    return fetcher

def update_prices(db, available_stocks, config, ensure_indexes=True):
    assert isinstance(config, dict)
    assert len(available_stocks) > 1000
  
    if ensure_indexes:
        db.asx_prices.create_index([('asx_code', pymongo.ASCENDING), ('fetch_date', pymongo.ASCENDING)], unique=True)

    fetcher = get_fetcher()
    df = None 
    fetch_date = datetime.now().strftime("%Y-%m-%d")
    for asx_code in available_stocks:
        url = "{}{}{}".format(config.get('asx_prices'), '' if config.get('asx_prices').endswith('/') else '/', asx_code)
        print("Fetching {} prices from {}".format(asx_code, url))
        already_fetched_doc = db.asx_prices.find_one({ 'asx_code': asx_code, 'fetch_date': fetch_date })
        if already_fetched_doc is not None:
            print("Already got data for ASX {}".format(asx_code))
            continue
        try:
            resp = fetcher.get(url)
            d = json.loads(resp.content.decode())
            d.update({ 'fetch_date': fetch_date })
            #assert len(d.keys()) > 10
            if df is None:
                df = pd.DataFrame(columns=d.keys()) 
            row = pd.Series(d, name=asx_code)
            df = df.append(row)
            db.asx_prices.find_one_and_update({ 'asx_code': asx_code, 'fetch_date': fetch_date }, { '$set': d }, upsert=True)
        except Exception as e:
            print("WARNING: unable to fetch data for {} -- ignored.".format(asx_code))
            print(str(e))
        time.sleep(5)  # be nice to the API endpoint
    fname = "{}/asx_prices.{}.tsv".format(config.get('data_root'), fetch_date)
    df.to_csv(fname, sep='\t')
    print("Saved {} stock codes with prices to {}".format(len(df), fname))
   
def available_stocks(db):
    ret = set([r.get('asx_code') for r in db.asx_isin.find({ 'security_name': 'ORDINARY FULLY PAID' }) 
                                              if db.asx_blacklist.find_one({'asx_code': r.get('asx_code') }) is None])
    print("Found {} available stocks on ASX...".format(len(ret)))
    return ret

def update_company_details(db, available_stocks, config, ensure_indexes=False):
    assert len(available_stocks) > 1000
    assert db is not None
    assert isinstance(config, dict)
    fetcher = get_fetcher() 

    if ensure_indexes:
        db.asx_company_details.create_index([ ('asx_code', pymongo.ASCENDING, ), ], unique=True)

    for asx_code in available_stocks:
        url = config.get('asx_company_details')
        url = url.replace('%s', asx_code)
        print(url)
        rec = db.asx_company_details.find_one({ 'asx_code': asx_code })
        if rec is not None:
            dt = rec.get('_id').generation_time
            if datetime.now(timezone.utc) < dt + timedelta(days=7):  # existing record less than a week old? if so, ignore it
                print("Ignoring {} as record is less than a week old.".format(asx_code))
                continue
        resp = fetcher.get(url)
        try:
            d = resp.json()
            d.update({ 'asx_code': asx_code })
            if 'error_desc' in d:
                print("WARNING: ignoring {} ASX code as it is not a valid code".format(asx_code))
                print(d) 
            else:
                assert d.pop('code', None) == asx_code
                db.asx_company_details.find_one_and_update({ 'asx_code': asx_code }, { '$set': d }, upsert=True)
        except Exception as e:
            print(str(e))
            pass 
        time.sleep(5)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Fetch and update data from ASX Research")
    args.add_argument('--config', help="Configuration file to use [config.json]", type=str, default="config.json")
    args.add_argument('--want-companies', help="Update companies list", action="store_true")
    args.add_argument('--want-isin', help="Update securities list", action="store_true")
    args.add_argument('--want-prices', help="Update ASX stock price list with current data", action="store_true")
    args.add_argument('--want-details', help="Update ASX company details (incl. dividend, annual report etc.) with current data", action="store_true")
    a = args.parse_args() 
 
    config = {} 
    with open(a.config, 'r') as fp:
        config = json.loads(fp.read())
    m = config.get('mongo')
    print(m)
    password = m.get('password')
    if password.startswith('$'):
        password = os.getenv(password[1:])
    mongo = pymongo.MongoClient(m.get('host'), m.get('port'), username=m.get('user'), password=password)
    db = mongo[m.get('db')]

    if a.want_companies:
        print("**** UPDATING ASX COMPANIES")
        update_companies(db, config, ensure_indexes=True) 
    if a.want_isin:
        print("**** UPDATING ASX SECURITIES")
        update_isin(db, config, ensure_indexes=True) 
    stocks_to_fetch = available_stocks(db)
    if a.want_prices:
        print("**** UPDATING PRICES")
        update_prices(db, stocks_to_fetch, config, ensure_indexes=True)
    if a.want_details:
        print("**** UPDATING COMPANY DETAILS")
        update_company_details(db, stocks_to_fetch, config, ensure_indexes=True)
         
    mongo.close()
    print("Run completed.")
    exit(0)
