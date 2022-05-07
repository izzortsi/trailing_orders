#!/usr/bin/env python

# %%

import logging
from binance.futures import Futures as Client
from binance.lib.utils import config_logging
from binance.error import ClientError

config_logging(logging, logging.DEBUG)

import os
key = os.environ.get('API_KEY')
sec = os.environ.get('API_SECRET')

client = Client(key, sec, base_url="https://fapi.binance.com")

try:
    response = client.new_order(symbol="ALGOUSDT", side = "BUY", type= "TAKE_PROFIT", quantity= 12.2, timeInForce="GTC", price = 0.7183, stopPrice = 0.7190)
    logging.info(response)
except ClientError as error:
    logging.error(
        "Found error. status: {}, error code: {}, error message: {}".format(
            error.status_code, error.error_code, error.error_message
        )
    )

#%%
