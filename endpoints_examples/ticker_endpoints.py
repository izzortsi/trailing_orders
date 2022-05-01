
# %%

import logging
from binance.futures import Futures as Client
from binance.lib.utils import config_logging

config_logging(logging, logging.DEBUG)

futures_client = Client()

logging.info(futures_client.ticker_24hr_price_change("BTCUSDT"))
#%%

logging.info(futures_client.ticker_price("BTCUSDT"))
#%%

logging.info(futures_client.book_ticker("BTCUSDT"))
#%%
