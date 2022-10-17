# %%
import logging
from constants import key, sec
from binance.futures import Futures as Client
from binance.lib.utils import config_logging
from binance.error import ClientError

# %%

config_logging(logging, logging.DEBUG)

client = Client(key, sec, base_url="https://fapi.binance.com")
try:
    # response = client.get_position_risk(symbol = "ENSUSDT", recvWindow=6000)
    response = client.get_position_risk(recvWindow=6000)
    logging.info(response)
except ClientError as error:
    logging.error(
        "Found error. status: {}, error code: {}, error message: {}".format(
            error.status_code, error.error_code, error.error_message
        )
    )
#%%
