#!/usr/bin/python3.8
import pymongo
import argparse
import requests
from datetime import datetime
import json
import csv
import olefile
import tempfile
import pandas as pd
import os
 
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
          
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Fetch and update data from ASX Research")
    args.add_argument('--config', help="Configuration file to use [config.json]", type=str, default="config.json")
    args.add_argument('--want-companies', help="Update companies list", action="store_true")
    args.add_argument('--want-isin', help="Update securities list", action="store_true")
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

    mongo.close()
    exit(0)
