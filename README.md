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

 ## Disclaimer

This software is provided to you "as-is" and without
warranty of any kind, express, implied or otherwise, including without
limitation, any warranty of fitness for a particular purpose. In no event shall
the author(s) be liable to you or anyone else for any direct, special, incidental,
indirect or consequential damages of any kind, or any damages whatsoever,
including without limitation, loss of profit, loss of use, savings or revenue,
or the claims of third parties, whether or not the author(s) have
been advised of the possibility of such loss, however caused and on any theory
of liability, arising out of or in connection with the possession, use or
performance of this software.


 ## System Requirements

  * Python 3.8

  * Django

  * Djongo (python MongoDB for Django apps)

  * Other requirements documented in [requirements.txt](./requirements.txt)


## Installation

### Pre-requisite software installation

On the system to run the website, you'll need to install required software:
~~~~
git clone https://github.com/ozacas/asxtrade.git
cd asxtrade
sudo pip3 install -r requirements.txt
cd src/viewer # for setting up the website
~~~~

### Setup MongoDB database server

Next, you'll want to setup a database to store the ASX data: instructions for this can be found at the [MongoDB website](https://docs.mongodb.com/manual/administration/install-community/)

### Setup the website and superuser for administrating the site

~~~~
python3 manage.py migrate

python3 manage.py createsuperuser

python3 manage.py runserver # run on local dev. host
~~~~

### Installing data

  You can run `python3 src/asxtrade.py --want-prices` to fetch daily data. This application only works with daily data fetched after 4pm each trading day from the ASX website. It will take several hours per run.

  Existing data ready to import into mongodb v4.4 can be fetched from [github large file storage](https://github.com/ozacas/asxtrade/raw/master/data/asxtrade.20210306.bson.gz) using [mongorestore](https://docs.mongodb.com/database-tools/mongorestore/). This data covers the daily data from July 2020 thru March 2020, although ETF data covers a smaller period due to missing code.

## Features

 | Feature             | Thumbnail Picture |
 |:--------------------|------------------:|
 | Portfolio watchlist | ![Pic](https://user-images.githubusercontent.com/11968760/91777314-da1bdb00-ec32-11ea-929e-66a1befc0d90.png#thumbnail)|
 | Stock view | ![Pic](https://user-images.githubusercontent.com/11968760/91777703-ed7b7600-ec33-11ea-87bf-b647033ed06f.png)|
 | Market sentiment | ![Pic](https://user-images.githubusercontent.com/11968760/91778464-e48ba400-ec35-11ea-9b47-413601da6fd8.png#thumbnail)|
 | Performance by sector | ![Pic](https://user-images.githubusercontent.com/11968760/110228446-6a760800-7f55-11eb-9041-786e6d145817.png#thumbnail)|
 | Portfolio optimisation | ![Pic](https://user-images.githubusercontent.com/11968760/110228663-e7ee4800-7f56-11eb-8b7d-edd3a09d7b29.png#thumbnail)

