
# %%

import logging
from binance.futures import Futures as Client
from binance.lib.utils import config_logging

config_logging(logging, logging.DEBUG)

futures_client = Client()

# logging.info(futures_client.klines("BTCUSDT", "1d"))
#%%
klines = futures_client.klines("BTCUSDT", "1d")
#%%
len(klines)
#%%