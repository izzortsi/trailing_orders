# %%

import logging
from binance.futures import Futures as Client
from binance.lib.utils import config_logging

config_logging(logging, logging.DEBUG)

futures_client = Client()

# logging.info(futures_client.index_info())
idx_info = futures_client.index_info()
#%%
for idx_symbol in idx_info:
    print(idx_symbol["symbol"])
#%%
