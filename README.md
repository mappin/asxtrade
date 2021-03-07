# asxtrade

Python3 based ASX data download and web application with basic features:

 * ability to search by sector, keyword, movement, dividend yield or other attributes

 * watchlist and sector stock lists, sortable by key metrics (eps, pe, daily change etc.)

 * graphs of datasets over 12 months (or whatever data is available)

 * HRP portfolio optimisation using [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html)

 * Market and per-sector performance incl. top 20/bottom 20

 * Virtual portfolios with trend visualisations, profit/loss by stock

 * Stock point-scoring by rules over time, based on their price performance

 * Visualisations provided by [plotnine](https://github.com/has2k1/plotnine) and [matplotlib](https://github.com/matplotlib/matplotlib)

 ## System Requirements

  * Python 3.8

  * Django

  * Djongo (python MongoDB for Django apps)

  * Other requirements documented in [requirements.txt](./requirements.txt)


 ## Installation

~~~~
sudo pip3 install -r requirements.txt
cd src/viewer
# setup mongodb and create a new database eg. asxtrade
# will be system specific eg. ubuntu
sudo apt-get install mongo-db
# setup newly installed database
# adjust settings.py eg. DATABASE to point to the new DB and...
python3 manage.py migrate

python3 manage.py createsuperuser
~~~~

 ## Installing data

  You can run `python3 src/asxtrade.py --want-prices` to fetch daily data. This application only works with daily data fetched after 4pm each trading day from the ASX website. It will take several hours per run.

  Existing data ready to import into mongodb v4.4 can be fetched from [github large file storage](https://github.com/ozacas/asxtrade/raw/master/data/asxtrade.20210306.bson.gz) using [mongorestore](https://docs.mongodb.com/database-tools/mongorestore/). This data covers the daily data from July 2020 thru March 2020, although ETF data covers a smaller period due to missing code.

 ## Features

 | Feature             | Thumbnail Picture |
 |:--------------------|------------------:|
 | Portfolio watchlist | ![Pic](https://user-images.githubusercontent.com/11968760/91777314-da1bdb00-ec32-11ea-929e-66a1befc0d90.png#thumbnail)|
 | Stock view | ![Pic](https://user-images.githubusercontent.com/11968760/91777703-ed7b7600-ec33-11ea-87bf-b647033ed06f.png)|
 | Market sentiment | ![Pic](https://user-images.githubusercontent.com/11968760/91778464-e48ba400-ec35-11ea-9b47-413601da6fd8.png)|

