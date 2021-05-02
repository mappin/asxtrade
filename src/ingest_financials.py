#!/usr/bin/python3
import pymongo
import argparse
import yfinance as yf
import time
from utils import read_config
import pandas as pd
from datetime import datetime, timedelta
from bson.objectid import ObjectId


def melt_dataframes(dfs: tuple) -> pd.DataFrame:
    result = None
    for df in filter(lambda df: df is not None and len(df) > 0, dfs):
        df["metric"] = df.index
        melted = pd.melt(df, id_vars=("metric"), var_name="date")
        melted = melted.dropna(axis=0, how="any")
        if len(melted) == 0:
            continue
        # print(melted)
        # print(melted.shape)
        if result is None:
            result = melted
        else:
            result = result.append(melted)
    if result is not None and "date" in result.columns:
        # print(result)
        result["date"] = pd.to_datetime(
            result["date"], infer_datetime_format=True
        )  # format="%Y-%m-%d")
    # print(result)
    return result


def desired_stocks():
    available_stocks = set(db.asx_company_details.distinct("asx_code"))
    gen_time = datetime.today() - timedelta(days=30)
    month_ago = ObjectId.from_datetime(gen_time)
    recently_updated_stocks = set(
        [
            rec["asx_code"]
            for rec in db.asx_company_financial_metrics.find(
                {"_id": {"$gte": month_ago}}
            )
        ]
    )

    return available_stocks.difference(recently_updated_stocks)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Update financial performance metrics for ASX stocks using yfinance"
    )
    args.add_argument(
        "--config",
        help="Configuration file to use [config.json]",
        type=str,
        default="config.json",
    )
    args.add_argument("--fail-fast", help="Stop on first error", action="store_true")
    args.add_argument(
        "--delay", help="Delay between stocks in seconds [30]", type=int, default=30
    )
    args.add_argument(
        "--debug",
        help="Try to fetch specified stock (for debugging)",
        type=str,
        required=False,
        default=None,
    )
    a = args.parse_args()
    config, password = read_config(a.config)
    m = config.get("mongo")
    mongo = pymongo.MongoClient(
        m.get("host"), m.get("port"), username=m.get("user"), password=password
    )
    db = mongo[m.get("db")]

    stock_codes = desired_stocks() if not a.debug else set([a.debug])
    print(f"Updating financial metrics for {len(stock_codes)} stocks")
    for asx_code in stock_codes:
        try:
            ticker = yf.Ticker(asx_code + ".AX")
            cashflow_df = ticker.cashflow
            financial_df = ticker.financials
            earnings_df = ticker.earnings
            if set(earnings_df.columns) == set(["Earnings", "Revenue"]):
                earnings_df.index = earnings_df.index.map(
                    str
                )  # convert years to str (maybe int)
                earnings_df = earnings_df.transpose()

            # print(earnings_df)
            balance_sheet_df = ticker.balance_sheet
            melted_df = melt_dataframes(
                (cashflow_df, financial_df, earnings_df, balance_sheet_df)
            )
            if melted_df is None or len(melted_df) < 1:
                raise ValueError(f"No data availale for {asx_code}... skipping")
            melted_df["asx_code"] = asx_code
            print(f"Updating {len(melted_df)} records for {asx_code}")
            for t in melted_df.itertuples():
                d = {
                    "metric": t.metric,
                    "date": t.date,
                    "value": t.value,
                    "asx_code": t.asx_code,
                }
                result = db.asx_company_financial_metrics.update_one(
                    {"asx_code": asx_code, "date": t.date, "metric": t.metric},
                    {"$set": d},
                    upsert=True,
                )
                assert result is not None
                assert isinstance(result, pymongo.results.UpdateResult)
                assert result.matched_count == 1 or result.upserted_id is not None
            time.sleep(a.delay)
        except Exception as e:
            print(f"WARNING: unable to download financials for {asx_code}")
            print(str(e))
            if a.fail_fast:
                raise e

    exit(0)
